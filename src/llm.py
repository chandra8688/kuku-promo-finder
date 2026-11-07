"""Centralized LLM Gateway (src/llm.py).

Rule (2): Every LLM call across the project must route through this module exclusively.
Provides unified rate limiting, exponential backoff with jitter, on-disk caching,
schema-enforced JSON with single repair retry, token/cost accounting, and concurrent batching.

Supports providers:
1. 'gemini' (native Google Gemini REST API v1beta)
2. 'openai_compatible' (any standard OpenAI-compatible endpoint)
3. 'mock' (offline scripted responses for hermetic unit testing)
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zoneinfo
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import yaml

try:
    TZ_LA = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    TZ_LA = timezone(timedelta(hours=-8))

try:
    TZ_IST = zoneinfo.ZoneInfo("Asia/Kolkata")
except Exception:
    TZ_IST = timezone(timedelta(hours=5, minutes=30))

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Data Models & Results
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    text: str
    parsed: Optional[Any]
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool
    attempts: int
    thinking_tokens: int = 0
    estimated_tokens: bool = False
    model_version_returned: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class LLMError(Exception):
    """Base error for LLM gateway failures."""
    pass


class LLMUnavailableError(LLMError):
    """Service unavailable or HTTP 503 error."""
    pass


class ModelUnavailableError(LLMError):
    """All models in fallback chain failed on 503/UNAVAILABLE."""
    pass


MODEL_UNAVAILABLE = ModelUnavailableError


class LLMAuthError(LLMError):
    """Authentication or permission error (HTTP 401 / 403)."""
    pass


class LLMBadRequestError(LLMError):
    """Client error (HTTP 400)."""
    pass


class LLMQuotaError(LLMError):
    """Rate limit or quota exhaustion (HTTP 429)."""
    pass


class LLMQuotaExhaustedError(LLMError):
    """Daily quota or credit limit exhausted (HTTP 429 RESOURCE_EXHAUSTED). Do not retry."""
    pass


class LLMRepairFailedError(LLMError):
    """Structured output failed schema validation even after repair retry."""
    pass


# ---------------------------------------------------------------------------
# Token Estimator & Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count at 4 characters per token."""
    return max(1, len(text) // 4)


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_GIT_COMMIT"


def get_config_hash(config_path: str = "config/models.yaml") -> str:
    """Calculate SHA256 hash of configuration file."""
    try:
        with open(config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "UNKNOWN_CONFIG_HASH"


# ---------------------------------------------------------------------------
# Thread-Safe Token Bucket Rate Limiter
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Per-role thread-safe token bucket rate limiter ensuring RPM is never exceeded."""

    def __init__(
        self,
        rpm: int = 15,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.rpm = max(1, rpm)
        self.capacity = float(self.rpm)
        self.tokens = float(self.rpm)
        self.refill_rate = float(self.rpm) / 60.0  # tokens per second
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.last_refill = self.time_fn()
        self._lock = threading.Lock()

    def notify_backoff(self, wait_s: float = 0.0) -> None:
        """Reset rate limiter tokens after backoff sleep."""
        with self._lock:
            self.tokens = 0.0
            self.last_refill = self.time_fn()

    def acquire(self) -> float:
        """Acquire 1 token, sleeping if necessary. Thread-safe."""
        with self._lock:
            now = self.time_fn()
            elapsed = now - self.last_refill
            if elapsed > 0:
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now

            wait_time = 0.0
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.refill_rate
                self.sleep_fn(wait_time)
                self.tokens = 0.0
                self.last_refill = self.time_fn()
            else:
                self.tokens -= 1.0

            return wait_time


# ---------------------------------------------------------------------------
# Persistent Usage Ledger (runs/usage.json)
# ---------------------------------------------------------------------------

class UsageLedger:
    """Persistent ledger in runs/usage.json tracking daily call counts per model in America/Los_Angeles."""

    def __init__(
        self,
        ledger_path: str | Path = "runs/usage.json",
        runs_dir: str | Path = "runs",
        tz_la: Optional[Any] = None,
        tz_ist: Optional[Any] = None,
    ):
        self.ledger_path = Path(ledger_path)
        self.runs_dir = Path(runs_dir)
        self.tz_la = tz_la or TZ_LA
        self.tz_ist = tz_ist or TZ_IST
        self._lock = threading.Lock()
        self._ensure_loaded()

    def get_current_la_day(self, dt: Optional[datetime] = None) -> str:
        """Return current calendar day string (YYYY-MM-DD) in America/Los_Angeles."""
        now = dt or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now_la = now.astimezone(self.tz_la)
        return now_la.strftime("%Y-%m-%d")

    def get_next_reset_ist(self, dt: Optional[datetime] = None) -> str:
        """Return next LA midnight reset time converted to Asia/Kolkata (IST)."""
        now = dt or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now_la = now.astimezone(self.tz_la)
        next_midnight_la = (now_la + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        reset_ist = next_midnight_la.astimezone(self.tz_ist)
        return reset_ist.strftime("%Y-%m-%d %I:%M:%S %p %Z")

    def _ensure_loaded(self) -> None:
        """Load from runs/usage.json or bootstrap from runs/*/calls.jsonl if missing."""
        records = self._load_records()
        if not records:
            records = self._bootstrap_from_runs()
        self._save_records(records)

    def _bootstrap_from_runs(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if self.runs_dir.exists():
            for p in self.runs_dir.glob("*/calls.jsonl"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    rec = json.loads(line)
                                    if not rec.get("cached", False):
                                        records.append(rec)
                                except Exception:
                                    pass
                except Exception:
                    pass
        return records

    def record_call(
        self,
        model: str,
        success: bool,
        tokens_in: int = 0,
        tokens_out: int = 0,
        timestamp: Optional[str] = None,
        cached: bool = False,
        model_version_returned: Optional[str] = None,
    ) -> None:
        """Record a completed or failed call to the ledger. Cached calls are never counted."""
        if cached:
            return
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": ts,
            "model": model,
            "model_version_returned": model_version_returned,
            "success": success,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cached": False,
        }
        with self._lock:
            records = self._load_records()
            records.append(entry)
            self._save_records(records)

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("records", [])
        except Exception:
            return []

    def _save_records(
        self, records: List[Dict[str, Any]], current_dt: Optional[datetime] = None
    ) -> None:
        now_utc = current_dt or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        current_la_day = self.get_current_la_day(now_utc)

        # Retain records from the last 7 days
        cutoff_utc = now_utc - timedelta(days=7)
        active_records: List[Dict[str, Any]] = []
        days_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

        for r in records:
            if r.get("cached", False):
                continue
            ts_str = r.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts >= cutoff_utc:
                active_records.append(r)
                r_day = ts.astimezone(self.tz_la).strftime("%Y-%m-%d")
                if r_day not in days_stats:
                    days_stats[r_day] = {}
                m = r.get("model_version_returned") or r.get("model", "unknown")
                if m not in days_stats[r_day]:
                    days_stats[r_day][m] = {
                        "successful_calls": 0,
                        "failed_calls": 0,
                        "total_calls": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                    }
                if r.get("success", True):
                    days_stats[r_day][m]["successful_calls"] += 1
                else:
                    days_stats[r_day][m]["failed_calls"] += 1
                days_stats[r_day][m]["total_calls"] += 1
                days_stats[r_day][m]["tokens_in"] += r.get("tokens_in", 0)
                days_stats[r_day][m]["tokens_out"] += r.get("tokens_out", 0)

        current_day_stats = days_stats.get(current_la_day, {})

        payload = {
            "updated_at": now_utc.isoformat(),
            "timezone": "America/Los_Angeles",
            "current_day_la": current_la_day,
            "next_reset_ist": self.get_next_reset_ist(now_utc),
            "models": current_day_stats,
            "days": days_stats,
            "records": active_records,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def get_stats_24h(
        self, model: Optional[str] = None, dt: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Return daily stats for the current LA day (or specified dt)."""
        with self._lock:
            records = self._load_records()
            self._save_records(records, current_dt=dt)
            if self.ledger_path.exists():
                try:
                    with open(self.ledger_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    models = data.get("models", {})
                    target_day = self.get_current_la_day(dt)
                    day_models = data.get("days", {}).get(target_day, {})
                    if model:
                        if model in day_models:
                            return day_models[model]
                        # Check records timestamped on target_day matching requested model
                        day_records = []
                        for r in data.get("records", []):
                            ts_str = r.get("timestamp")
                            if ts_str:
                                try:
                                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                    if ts.tzinfo is None:
                                        ts = ts.replace(tzinfo=timezone.utc)
                                    if ts.astimezone(self.tz_la).strftime("%Y-%m-%d") == target_day:
                                        if r.get("model") == model or r.get("model_version_returned") == model:
                                            day_records.append(r)
                                except Exception:
                                    pass
                        if day_records:
                            succ = sum(1 for r in day_records if r.get("success", True))
                            fail = sum(1 for r in day_records if not r.get("success", True))
                            tin = sum(r.get("tokens_in", 0) for r in day_records)
                            tout = sum(r.get("tokens_out", 0) for r in day_records)
                            return {
                                "successful_calls": succ,
                                "failed_calls": fail,
                                "total_calls": succ + fail,
                                "tokens_in": tin,
                                "tokens_out": tout,
                            }
                        return {
                            "successful_calls": 0,
                            "failed_calls": 0,
                            "total_calls": 0,
                            "tokens_in": 0,
                            "tokens_out": 0,
                        }
                    return day_models
                except Exception:
                    pass
            return {}

    def get_remaining_daily_quota(
        self, model: str, rpd_limit: Optional[int], dt: Optional[datetime] = None
    ) -> Optional[int]:
        """Return remaining calls for a model on current LA day, or None if no limit."""
        if rpd_limit is None:
            return None
        stats = self.get_stats_24h(model, dt=dt)
        used = stats.get("total_calls", 0)
        return max(0, rpd_limit - used)


# ---------------------------------------------------------------------------
# Model Gateway
# ---------------------------------------------------------------------------

class ModelGateway:
    """Central gateway handling all LLM calls, caching, rate limiting, and metrics."""

    def __init__(
        self,
        models_config_path: str = "config/models.yaml",
        roles_config: Optional[Dict[str, Dict[str, Any]]] = None,
        cache_dir: str = "cache",
        runs_dir: str = "runs",
        enable_cache: bool = True,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        load_dotenv(".env")
        self.models_config_path = models_config_path
        self.cache_dir = Path(cache_dir)
        self.runs_dir = Path(runs_dir)
        self.enable_cache = enable_cache
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.roles_config = (
            roles_config if roles_config is not None else self._load_roles_config()
        )
        self.rate_limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._init_rate_limiters()

        # Persistent usage ledger
        self.usage_ledger = UsageLedger(
            ledger_path=self.runs_dir / "usage.json", runs_dir=self.runs_dir
        )

        # Thread synchronization locks
        self._accounting_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._mock_lock = threading.Lock()

        # Mock responses queue for testing: { (role, prompt_keyword): [responses/exceptions] }
        self.mock_registry: Dict[str, List[Any]] = {}

        # Default run tracking
        self.default_run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    def _load_roles_config(self) -> Dict[str, Dict[str, Any]]:
        path = Path(self.models_config_path)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("roles", {})
        except Exception as e:
            raise LLMError(f"Failed to load models config from {path}: {e}")

    def _init_rate_limiters(self) -> None:
        for role, cfg in self.roles_config.items():
            rpm = cfg.get("rpm_limit", 15)
            self.rate_limiters[role] = TokenBucketRateLimiter(
                rpm=rpm,
                time_fn=self.time_fn,
                sleep_fn=self.sleep_fn,
            )

    def register_mock_response(self, role: str, response_or_exception: Any) -> None:
        """Register a scripted response or Exception for offline unit testing."""
        with self._mock_lock:
            self.mock_registry.setdefault(role, []).append(response_or_exception)

    def _compute_cache_key(
        self,
        stage: str,
        prompt_version: str,
        provider: str,
        model_name: str,
        temperature: float,
        system: str,
        user: str,
        schema_name: Optional[str],
        cache_key_extra: Optional[str],
    ) -> str:
        payload = json.dumps(
            {
                "stage": stage,
                "prompt_version": prompt_version,
                "provider": provider,
                "model_name": model_name,
                "temperature": temperature,
                "system": system,
                "user": user,
                "schema_name": schema_name,
                "cache_key_extra": cache_key_extra,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _append_accounting(
        self,
        run_id: str,
        stage: str,
        role: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        thinking_tokens: int,
        latency_s: float,
        cached: bool,
        attempts: int,
        success: bool,
        estimated_tokens: bool,
        model_version_returned: Optional[str] = None,
    ) -> None:
        """Appends accounting entry to runs/<run_id>/calls.jsonl. Thread-safe. NEVER logs text!"""
        run_folder = self.runs_dir / run_id
        run_folder.mkdir(parents=True, exist_ok=True)
        calls_file = run_folder / "calls.jsonl"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "role": role,
            "provider": provider,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "thinking_tokens": thinking_tokens,
            "latency_s": round(latency_s, 4),
            "cached": cached,
            "attempts": attempts,
            "success": success,
            "estimated_tokens": estimated_tokens,
            "model_version_returned": model_version_returned,
        }

        with self._accounting_lock:
            with open(calls_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if hasattr(self, "usage_ledger") and self.usage_ledger and not cached:
                self.usage_ledger.record_call(
                    model=model,
                    success=success,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    timestamp=entry["timestamp"],
                    cached=cached,
                    model_version_returned=model_version_returned,
                )

    def write_manifest(self, run_id: Optional[str] = None) -> Path:
        """Write or update manifest.json for a run."""
        rid = run_id or self.default_run_id
        run_folder = self.runs_dir / rid
        run_folder.mkdir(parents=True, exist_ok=True)
        manifest_file = run_folder / "manifest.json"
        calls_file = run_folder / "calls.jsonl"

        total_calls = 0
        cached_calls = 0
        total_in = 0
        total_out = 0
        total_thinking = 0
        total_lat = 0.0
        models_used = set()
        providers_used = set()
        model_versions_returned = set()

        if calls_file.exists():
            with open(calls_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        c = json.loads(line)
                        total_calls += 1
                        if c.get("cached"):
                            cached_calls += 1
                        total_in += c.get("tokens_in", 0)
                        total_out += c.get("tokens_out", 0)
                        total_thinking += c.get("thinking_tokens", 0)
                        total_lat += c.get("latency_s", 0.0)
                        models_used.add(c.get("model", ""))
                        providers_used.add(c.get("provider", ""))
                        mv = c.get("model_version_returned")
                        if mv:
                            model_versions_returned.add(mv)
                    except Exception:
                        pass

        manifest_data = {
            "run_id": rid,
            "git_commit": get_git_commit(),
            "config_hash": get_config_hash(self.models_config_path),
            "providers_used": sorted(list(providers_used)),
            "models_used": sorted(list(models_used)),
            "model_versions_returned": sorted(list(model_versions_returned)),
            "safety_warning": (
                "NOTE ON FREE-TIER PRIVACY: When using free-tier providers (such as Google "
                "Gemini free tier), prompts and responses may be logged by the provider for "
                "model improvement. For confidential production microdramas, use enterprise "
                "or self-hosted openai_compatible endpoints."
            ),
            "total_calls": total_calls,
            "cached_calls": cached_calls,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_thinking_tokens": total_thinking,
            "total_latency_s": round(total_lat, 2),
        }

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest_file

    def call(
        self,
        role: str,
        system: str,
        user: str,
        schema: Optional[Type[T]] = None,
        stage: str = "default",
        prompt_version: str = "v1",
        cache_key_extra: Optional[str] = None,
        enable_cache: Optional[bool] = None,
        run_id: Optional[str] = None,
    ) -> LLMResult:
        """Execute model call through gateway with caching, rate limits, and retries."""
        active_run_id = run_id or self.default_run_id

        # Fetch role config
        cfg = self.roles_config.get(role)
        if not cfg:
            cfg = {
                "provider": "mock",
                "model_name": "mock-model",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "rpm_limit": 15,
                "api_key_env": "GEMINI_API_KEY",
            }

        provider = cfg.get("provider", "gemini").lower()
        model_name = cfg.get("model_name", "gemini-3.5-flash")
        temperature = float(cfg.get("temperature", 0.0))
        max_output_tokens = int(cfg.get("max_output_tokens", 2048))
        thinking_budget = cfg.get("thinking_budget")
        api_key_env = cfg.get("api_key_env", "GEMINI_API_KEY")
        base_url = cfg.get("base_url")

        # Resolve API Key
        api_key = os.environ.get(api_key_env, "").strip()
        if provider != "mock" and not api_key:
            raise LLMAuthError(
                f"Missing API key: Environment variable '{api_key_env}' is not set in .env."
            )

        # Cache check
        use_cache = self.enable_cache if enable_cache is None else enable_cache
        schema_name = schema.__name__ if schema else None
        cache_key = self._compute_cache_key(
            stage=stage,
            prompt_version=prompt_version,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            system=system,
            user=user,
            schema_name=schema_name,
            cache_key_extra=cache_key_extra,
        )
        cache_file = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                parsed_val = None
                if schema and cached_data.get("text"):
                    parsed_val = schema.model_validate_json(cached_data["text"])

                # Accounting for cache hit
                cached_mv = cached_data.get("model_version_returned")
                self._append_accounting(
                    run_id=active_run_id,
                    stage=stage,
                    role=role,
                    provider=provider,
                    model=model_name,
                    tokens_in=cached_data.get("input_tokens", 0),
                    tokens_out=cached_data.get("output_tokens", 0),
                    thinking_tokens=cached_data.get("thinking_tokens", 0),
                    latency_s=0.0,
                    cached=True,
                    attempts=1,
                    success=True,
                    estimated_tokens=cached_data.get("estimated_tokens", False),
                    model_version_returned=cached_mv,
                )

                return LLMResult(
                    text=cached_data["text"],
                    parsed=parsed_val,
                    model=model_name,
                    provider=provider,
                    input_tokens=cached_data.get("input_tokens", 0),
                    output_tokens=cached_data.get("output_tokens", 0),
                    thinking_tokens=cached_data.get("thinking_tokens", 0),
                    latency_s=0.0,
                    cached=True,
                    attempts=1,
                    estimated_tokens=cached_data.get("estimated_tokens", False),
                    model_version_returned=cached_mv,
                )
            except Exception:
                pass

        # Build fallback model chain for role
        model_chain: List[Dict[str, Any]] = [dict(cfg)]
        fallback_roles = cfg.get("fallback_roles")
        if fallback_roles is None and role == "cheap":
            fallback_roles = ["judge"]
        elif fallback_roles is None:
            fallback_roles = []

        for fb_role in fallback_roles:
            if fb_role in self.roles_config:
                fb_cfg = dict(self.roles_config[fb_role])
                if fb_cfg.get("model_name") != cfg.get("model_name"):
                    model_chain.append(fb_cfg)

        if "fallback_models" in cfg:
            for fb_m in cfg["fallback_models"]:
                if isinstance(fb_m, dict):
                    model_chain.append(fb_m)
                elif isinstance(fb_m, str):
                    model_chain.append({
                        "provider": cfg.get("provider", "mock"),
                        "model_name": fb_m,
                        "temperature": cfg.get("temperature", 0.0),
                        "max_output_tokens": cfg.get("max_output_tokens", 2048),
                        "rpm_limit": cfg.get("rpm_limit", 15),
                        "api_key_env": cfg.get("api_key_env", "GEMINI_API_KEY"),
                    })

        total_attempts = 0
        start_time = self.time_fn()
        raw_text = ""
        in_tokens = 0
        out_tokens = 0
        thinking_tokens = 0
        is_estimated = False
        model_version_returned = None
        succeeded = False
        last_error = None
        active_model_cfg = cfg

        for m_cfg in model_chain:
            active_model_cfg = m_cfg
            provider = m_cfg.get("provider", "gemini").lower()
            model_name = m_cfg.get("model_name", "gemini-3.5-flash")
            temperature = float(m_cfg.get("temperature", 0.0))
            max_output_tokens = int(m_cfg.get("max_output_tokens", 2048))
            thinking_budget = m_cfg.get("thinking_budget")
            api_key_env = m_cfg.get("api_key_env", "GEMINI_API_KEY")
            base_url = m_cfg.get("base_url")

            # Resolve API Key
            api_key = os.environ.get(api_key_env, "").strip()
            if provider != "mock" and not api_key:
                raise LLMAuthError(
                    f"Missing API key: Environment variable '{api_key_env}' is not set in .env."
                )

            rate_limiter = self.rate_limiters.get(model_name) or self.rate_limiters.get(role)
            if not rate_limiter:
                rpm = m_cfg.get("rpm_limit", 15)
                rate_limiter = TokenBucketRateLimiter(
                    rpm=rpm, time_fn=self.time_fn, sleep_fn=self.sleep_fn
                )
                self.rate_limiters[model_name] = rate_limiter

            model_attempts = 0
            max_503_attempts = 3
            max_retries = 10

            while model_attempts < max_retries:
                model_attempts += 1
                total_attempts += 1
                rate_limiter.acquire()

                try:
                    if provider == "mock":
                        raw_text, in_tokens, out_tokens, thinking_tokens, is_estimated, model_version_returned = self._call_mock(
                            role, system, user, schema, model_name=model_name
                        )
                    elif provider == "gemini":
                        raw_text, in_tokens, out_tokens, thinking_tokens, is_estimated, model_version_returned = self._call_gemini(
                            model=model_name,
                            api_key=api_key,
                            system=system,
                            user=user,
                            temperature=temperature,
                            max_tokens=max_output_tokens,
                            thinking_budget=thinking_budget,
                            schema=schema,
                        )
                    elif provider == "openai_compatible":
                        raw_text, in_tokens, out_tokens, thinking_tokens, is_estimated, model_version_returned = self._call_openai(
                            model=model_name,
                            api_key=api_key,
                            base_url=base_url or "https://api.openai.com/v1",
                            system=system,
                            user=user,
                            temperature=temperature,
                            max_tokens=max_output_tokens,
                            schema=schema,
                        )
                    else:
                        raise LLMError(f"Unsupported provider: '{provider}'")

                    # Successful network call on active model
                    succeeded = True
                    break

                except (LLMAuthError, LLMBadRequestError, LLMQuotaExhaustedError) as e:
                    # Fatal errors or daily quota exhaustion: never retry
                    self._append_accounting(
                        run_id=active_run_id,
                        stage=stage,
                        role=role,
                        provider=provider,
                        model=model_name,
                        tokens_in=0,
                        tokens_out=0,
                        thinking_tokens=0,
                        latency_s=self.time_fn() - start_time,
                        cached=False,
                        attempts=total_attempts,
                        success=False,
                        estimated_tokens=False,
                        model_version_returned=None,
                    )
                    raise e

                except LLMUnavailableError as e:
                    last_error = e
                    # Cap retries on 503/UNAVAILABLE at 3 attempts with backoff up to 30s
                    if model_attempts >= max_503_attempts:
                        self._append_accounting(
                            run_id=active_run_id,
                            stage=stage,
                            role=role,
                            provider=provider,
                            model=model_name,
                            tokens_in=0,
                            tokens_out=0,
                            thinking_tokens=0,
                            latency_s=self.time_fn() - start_time,
                            cached=False,
                            attempts=model_attempts,
                            success=False,
                            estimated_tokens=False,
                            model_version_returned=None,
                        )
                        break

                    retry_after = getattr(e, "retry_after", None)
                    backoff = min(30.0, (2 ** (model_attempts - 1)) + random.uniform(0.1, 0.5))
                    sleep_duration = min(30.0, max(float(retry_after) if retry_after else 0.0, backoff))
                    self.sleep_fn(sleep_duration)

                except (LLMQuotaError, LLMError, urllib.error.URLError, TimeoutError) as e:
                    last_error = e
                    err_str = str(e).upper()
                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        if model_attempts >= max_503_attempts:
                            self._append_accounting(
                                run_id=active_run_id,
                                stage=stage,
                                role=role,
                                provider=provider,
                                model=model_name,
                                tokens_in=0,
                                tokens_out=0,
                                thinking_tokens=0,
                                latency_s=self.time_fn() - start_time,
                                cached=False,
                                attempts=model_attempts,
                                success=False,
                                estimated_tokens=False,
                                model_version_returned=None,
                            )
                            break
                        retry_after = getattr(e, "retry_after", None)
                        backoff = min(30.0, (2 ** (model_attempts - 1)) + random.uniform(0.1, 0.5))
                        sleep_duration = min(30.0, max(float(retry_after) if retry_after else 0.0, backoff))
                        self.sleep_fn(sleep_duration)
                        continue

                    if model_attempts >= max_retries:
                        self._append_accounting(
                            run_id=active_run_id,
                            stage=stage,
                            role=role,
                            provider=provider,
                            model=model_name,
                            tokens_in=0,
                            tokens_out=0,
                            thinking_tokens=0,
                            latency_s=self.time_fn() - start_time,
                            cached=False,
                            attempts=total_attempts,
                            success=False,
                            estimated_tokens=False,
                            model_version_returned=None,
                        )
                        raise LLMError(f"Max retries ({max_retries}) exceeded for {role}: {e}")

                    retry_after = getattr(e, "retry_after", None)
                    backoff = min(60.0, (2 ** (model_attempts - 1)) + random.uniform(0.1, 0.5))
                    sleep_duration = max(float(retry_after) if retry_after else 0.0, backoff)
                    self.sleep_fn(sleep_duration)

            if succeeded:
                break

        if not succeeded:
            raise ModelUnavailableError(
                f"MODEL_UNAVAILABLE: All models in fallback chain failed for role '{role}': {last_error}"
            )

        attempts = total_attempts

        latency = round(self.time_fn() - start_time, 4)

        # Structured output validation & single repair retry
        parsed_result = None
        if schema:
            try:
                parsed_result = schema.model_validate_json(raw_text)
            except (ValidationError, json.JSONDecodeError) as val_err:
                repair_prompt = (
                    f"Your previous output was invalid according to the required JSON schema.\n"
                    f"Validation Error: {val_err}\n"
                    f"Invalid Output:\n{raw_text}\n"
                    f"Please provide the corrected valid JSON strictly adhering to the schema."
                )

                try:
                    attempts += 1
                    rate_limiter.acquire()
                    if provider == "mock":
                        repair_text, r_in, r_out, r_th, r_est, r_mv = self._call_mock(
                            role, system, repair_prompt, schema
                        )
                    elif provider == "gemini":
                        repair_text, r_in, r_out, r_th, r_est, r_mv = self._call_gemini(
                            model=model_name,
                            api_key=api_key,
                            system=system,
                            user=repair_prompt,
                            temperature=temperature,
                            max_tokens=max_output_tokens,
                            thinking_budget=thinking_budget,
                            schema=schema,
                        )
                    elif provider == "openai_compatible":
                        repair_text, r_in, r_out, r_th, r_est, r_mv = self._call_openai(
                            model=model_name,
                            api_key=api_key,
                            base_url=base_url or "https://api.openai.com/v1",
                            system=system,
                            user=repair_prompt,
                            temperature=temperature,
                            max_tokens=max_output_tokens,
                            schema=schema,
                        )
                    else:
                        repair_text = raw_text
                        r_in, r_out, r_th, r_est, r_mv = in_tokens, out_tokens, thinking_tokens, is_estimated, model_version_returned

                    in_tokens += r_in
                    out_tokens += r_out
                    thinking_tokens += r_th
                    raw_text = repair_text
                    if r_mv:
                        model_version_returned = r_mv
                    parsed_result = schema.model_validate_json(raw_text)
                except Exception as final_err:
                    self._append_accounting(
                        run_id=active_run_id,
                        stage=stage,
                        role=role,
                        provider=provider,
                        model=model_name,
                        tokens_in=in_tokens,
                        tokens_out=out_tokens,
                        thinking_tokens=thinking_tokens,
                        latency_s=round(self.time_fn() - start_time, 4),
                        cached=False,
                        attempts=attempts,
                        success=False,
                        estimated_tokens=is_estimated,
                        model_version_returned=model_version_returned,
                    )
                    raise LLMRepairFailedError(
                        f"Schema validation failed after repair retry: {final_err}"
                    )

        # Cache valid result
        if use_cache:
            try:
                cache_payload = {
                    "text": raw_text,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "thinking_tokens": thinking_tokens,
                    "estimated_tokens": is_estimated,
                    "model": model_name,
                    "provider": provider,
                    "model_version_returned": model_version_returned,
                }
                with self._cache_lock:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache_payload, f, indent=2)
            except Exception:
                pass

        # Accounting record
        self._append_accounting(
            run_id=active_run_id,
            stage=stage,
            role=role,
            provider=provider,
            model=model_name,
            tokens_in=in_tokens,
            tokens_out=out_tokens,
            thinking_tokens=thinking_tokens,
            latency_s=latency,
            cached=False,
            attempts=attempts,
            success=True,
            estimated_tokens=is_estimated,
            model_version_returned=model_version_returned,
        )

        return LLMResult(
            text=raw_text,
            parsed=parsed_result,
            model=model_name,
            provider=provider,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            thinking_tokens=thinking_tokens,
            latency_s=latency,
            cached=False,
            attempts=attempts,
            estimated_tokens=is_estimated,
            model_version_returned=model_version_returned,
        )

    def call_many(
        self,
        requests: List[Dict[str, Any]],
        max_workers: int = 4,
        run_id: Optional[str] = None,
    ) -> List[LLMResult]:
        """Execute multiple requests concurrently across a thread pool.

        Results are returned in the exact same order as requests.
        Rate limiter ensures the combined request rate never exceeds rpm_limit.
        """
        if not requests:
            return []

        results: List[Optional[LLMResult]] = [None] * len(requests)
        active_run_id = run_id or self.default_run_id

        def _worker(idx: int, req_kwargs: Dict[str, Any]) -> Tuple[int, LLMResult]:
            # Inject active_run_id if not present
            kwargs = dict(req_kwargs)
            if "run_id" not in kwargs:
                kwargs["run_id"] = active_run_id
            try:
                res = self.call(**kwargs)
                return idx, res
            except LLMQuotaExhaustedError:
                raise
            except Exception as e:
                role_val = kwargs.get("role", "unknown")
                cfg_val = self.roles_config.get(role_val, {})
                failed_res = LLMResult(
                    text="",
                    parsed=None,
                    model=cfg_val.get("model_name", role_val),
                    provider=cfg_val.get("provider", "unknown"),
                    input_tokens=0,
                    output_tokens=0,
                    latency_s=0.0,
                    cached=False,
                    attempts=1,
                    success=False,
                    error=str(e),
                )
                return idx, failed_res

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_worker, i, req) for i, req in enumerate(requests)
            ]
            exhausted_err = None
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, res = fut.result()
                    results[idx] = res
                except LLMQuotaExhaustedError as e:
                    exhausted_err = e
                    for f in futures:
                        f.cancel()
                    break

            if exhausted_err is not None:
                raise exhausted_err

        return [r for r in results if r is not None]

    def get_remaining_quota(self, role: str) -> Optional[int]:
        """Return remaining calls for role's model in rolling 24h, or None if no limit."""
        cfg = self.roles_config.get(role, {})
        model = cfg.get("model_name", "")
        rpd = cfg.get("rpd_limit")
        if rpd is None:
            return None
        return self.usage_ledger.get_remaining_daily_quota(model, rpd)

    def check_stage_quota(self, role: str, required_calls: int, force: bool = False) -> None:
        """Check if required_calls exceeds remaining 24h quota. Raises LLMQuotaExhaustedError if so, unless force=True."""
        rem = self.get_remaining_quota(role)
        if rem is not None and required_calls > rem:
            cfg = self.roles_config.get(role, {})
            model = cfg.get("model_name", role)
            msg = (
                f"Stage requires {required_calls} call(s) for role '{role}' ({model}), "
                f"but only {rem} call(s) remain in the rolling 24h quota (limit {cfg.get('rpd_limit')}). Refusing to start."
            )
            if not force:
                raise LLMQuotaExhaustedError(f"{msg} Use --force to proceed anyway.")

    # -----------------------------------------------------------------------
    # Provider Implementations
    # -----------------------------------------------------------------------

    def _call_gemini(
        self,
        model: str,
        api_key: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        thinking_budget: Optional[int],
        schema: Optional[Type[BaseModel]],
    ) -> Tuple[str, int, int, int, bool, Optional[str]]:
        """Call Google Gemini REST API v1beta."""
        clean_model = model.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"

        gen_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if schema:
            gen_config["responseMimeType"] = "application/json"

        # Apply thinkingBudget if specified
        if thinking_budget is not None:
            gen_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}

        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": gen_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "promo-finder/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            retry_after = None
            if e.headers and e.headers.get("Retry-After"):
                try:
                    retry_after = float(e.headers.get("Retry-After"))
                except Exception:
                    pass

            if e.code in (401, 403):
                raise LLMAuthError(f"Gemini Auth Error (HTTP {e.code}): {err_body}")
            elif e.code == 400:
                raise LLMBadRequestError(f"Gemini Client Error (HTTP 400): {err_body}")
            elif e.code == 429:
                if not retry_after:
                    try:
                        err_json = json.loads(err_body)
                        details = err_json.get("error", {}).get("details", [])
                        for d in details:
                            if "retryDelay" in d:
                                rd_str = str(d["retryDelay"]).rstrip("s")
                                retry_after = float(rd_str)
                                break
                    except Exception:
                        pass

                is_daily = bool(
                    re.search(r"(?i)\bper[- ]?day\b|generaterequestsperday|perday", err_body)
                )
                if is_daily:
                    raise LLMQuotaExhaustedError(f"Gemini daily quota exhausted (HTTP 429): {err_body}")
                err = LLMQuotaError(f"Gemini Quota/Rate Limit (HTTP 429): {err_body}")
                err.retry_after = retry_after
                raise err
            elif e.code == 503 or (e.code >= 500 and "UNAVAILABLE" in err_body.upper()):
                err = LLMUnavailableError(f"Gemini Server Unavailable (HTTP {e.code}): {err_body}")
                err.retry_after = retry_after
                err.status_code = e.code
                raise err
            elif e.code >= 500:
                err = LLMError(f"Gemini Server Error (HTTP {e.code}): {err_body}")
                err.retry_after = retry_after
                raise err
            else:
                raise LLMError(f"Gemini HTTP {e.code} Error: {err_body}")
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"Gemini Network/Timeout Error: {e}")

        candidates = data.get("candidates", [])
        if not candidates or "content" not in candidates[0]:
            raise LLMError("Gemini returned empty candidate response")

        parts = candidates[0]["content"].get("parts", [])
        raw_text = "".join(p.get("text", "") for p in parts).strip()

        # Extract tokens and thoughts
        usage = data.get("usageMetadata", {})
        in_tokens = usage.get("promptTokenCount")
        candidate_tokens = usage.get("candidatesTokenCount")
        thinking_tokens = usage.get("thoughtsTokenCount", 0)

        is_estimated = False
        if in_tokens is None or candidate_tokens is None:
            in_tokens = estimate_tokens(system + user)
            candidate_tokens = estimate_tokens(raw_text)
            is_estimated = True

        # Count reasoning tokens as part of output tokens
        total_out_tokens = candidate_tokens + thinking_tokens
        model_version_returned = data.get("modelVersion")

        return raw_text, in_tokens, total_out_tokens, thinking_tokens, is_estimated, model_version_returned

    def _call_openai(
        self,
        model: str,
        api_key: str,
        base_url: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        schema: Optional[Type[BaseModel]],
    ) -> Tuple[str, int, int, int, bool, Optional[str]]:
        """Call standard OpenAI-compatible chat completion endpoint."""
        url = f"{base_url.rstrip('/')}/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "promo-finder/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            retry_after = None
            if e.headers and e.headers.get("Retry-After"):
                try:
                    retry_after = float(e.headers.get("Retry-After"))
                except Exception:
                    pass

            if e.code in (401, 403):
                raise LLMAuthError(f"OpenAI Auth Error (HTTP {e.code}): {err_body}")
            elif e.code == 400:
                raise LLMBadRequestError(f"OpenAI Client Error (HTTP 400): {err_body}")
            elif e.code == 429:
                if not retry_after:
                    try:
                        err_json = json.loads(err_body)
                        if "retry_after" in err_json:
                            retry_after = float(err_json["retry_after"])
                    except Exception:
                        pass
                is_daily = bool(
                    re.search(r"(?i)\bper[- ]?day\b|perday|insufficient_quota", err_body)
                )
                if is_daily:
                    raise LLMQuotaExhaustedError(f"OpenAI quota exhausted (HTTP 429): {err_body}")
                err = LLMQuotaError(f"OpenAI Quota/Rate Limit (HTTP 429): {err_body}")
                err.retry_after = retry_after
                raise err
            elif e.code == 503 or (e.code >= 500 and "UNAVAILABLE" in err_body.upper()):
                err = LLMUnavailableError(f"OpenAI Server Unavailable (HTTP {e.code}): {err_body}")
                err.retry_after = retry_after
                err.status_code = e.code
                raise err
            elif e.code >= 500:
                err = LLMError(f"OpenAI Server Error (HTTP {e.code}): {err_body}")
                err.retry_after = retry_after
                raise err
            else:
                raise LLMError(f"OpenAI HTTP {e.code} Error: {err_body}")
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"OpenAI Network/Timeout Error: {e}")

        choices = data.get("choices", [])
        if not choices:
            raise LLMError("OpenAI returned empty choices array")

        raw_text = choices[0].get("message", {}).get("content", "").strip()

        usage = data.get("usage", {})
        in_tokens = usage.get("prompt_tokens")
        out_tokens = usage.get("completion_tokens")
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        is_estimated = False
        if in_tokens is None or out_tokens is None:
            in_tokens = estimate_tokens(system + user)
            out_tokens = estimate_tokens(raw_text)
            is_estimated = True

        model_version_returned = data.get("model")

        return raw_text, in_tokens, out_tokens, reasoning_tokens, is_estimated, model_version_returned

    def _call_mock(
        self,
        role: str,
        system: str,
        user: str,
        schema: Optional[Type[BaseModel]],
        model_name: Optional[str] = None,
    ) -> Tuple[str, int, int, int, bool, Optional[str]]:
        """Offline mock provider for testing."""
        with self._mock_lock:
            queue = None
            if model_name and model_name in self.mock_registry and self.mock_registry[model_name]:
                queue = self.mock_registry[model_name]
            elif role in self.mock_registry and self.mock_registry[role]:
                queue = self.mock_registry[role]
            else:
                queue = []

            if queue:
                item = queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, str):
                    raw_text = item
                elif isinstance(item, dict):
                    raw_text = json.dumps(item)
                elif isinstance(item, BaseModel):
                    raw_text = item.model_dump_json()
                else:
                    raw_text = str(item)
            else:
                if schema:
                    raw_text = '{"status": "ok", "score": 3, "reason": "mock valid"}'
                else:
                    raw_text = "OK"

        in_tokens = estimate_tokens(system + user)
        out_tokens = estimate_tokens(raw_text)
        return raw_text, in_tokens, out_tokens, 0, True, "mock-model"


# ---------------------------------------------------------------------------
# Module-level singleton gateway and convenience functions
# ---------------------------------------------------------------------------

_GLOBAL_GATEWAY: Optional[ModelGateway] = None


def get_gateway() -> ModelGateway:
    """Retrieve or initialize global ModelGateway instance."""
    global _GLOBAL_GATEWAY
    if _GLOBAL_GATEWAY is None:
        _GLOBAL_GATEWAY = ModelGateway()
    return _GLOBAL_GATEWAY


def set_gateway(gw: Optional[ModelGateway]) -> None:
    """Override or reset the global ModelGateway instance."""
    global _GLOBAL_GATEWAY
    _GLOBAL_GATEWAY = gw


def reset_gateway() -> None:
    """Reset the global ModelGateway instance to None."""
    global _GLOBAL_GATEWAY
    _GLOBAL_GATEWAY = None


def call(
    role: str,
    system: str,
    user: str,
    schema: Optional[Type[T]] = None,
    stage: str = "default",
    prompt_version: str = "v1",
    cache_key_extra: Optional[str] = None,
    enable_cache: Optional[bool] = None,
    run_id: Optional[str] = None,
) -> LLMResult:
    """Universal gateway interface."""
    return get_gateway().call(
        role=role,
        system=system,
        user=user,
        schema=schema,
        stage=stage,
        prompt_version=prompt_version,
        cache_key_extra=cache_key_extra,
        enable_cache=enable_cache,
        run_id=run_id,
    )


def call_many(
    requests: List[Dict[str, Any]],
    max_workers: int = 4,
    run_id: Optional[str] = None,
) -> List[LLMResult]:
    """Execute multiple requests concurrently via thread pool with rate-limiting."""
    return get_gateway().call_many(
        requests=requests,
        max_workers=max_workers,
        run_id=run_id,
    )


def get_remaining_quota(role: str) -> Optional[int]:
    """Return remaining calls for role's model in rolling 24h, or None if no limit."""
    return get_gateway().get_remaining_quota(role)


def check_stage_quota(role: str, required_calls: int, force: bool = False) -> None:
    """Check if required_calls exceeds remaining quota for role. Raises LLMQuotaExhaustedError if not forced."""
    get_gateway().check_stage_quota(role, required_calls, force=force)


def get_usage_ledger() -> UsageLedger:
    """Return the gateway's usage ledger."""
    return get_gateway().usage_ledger
