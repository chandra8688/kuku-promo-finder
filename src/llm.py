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
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import yaml

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


class LLMError(Exception):
    """Base error for LLM gateway failures."""
    pass


class LLMAuthError(LLMError):
    """Authentication or permission error (HTTP 401 / 403)."""
    pass


class LLMBadRequestError(LLMError):
    """Client error (HTTP 400)."""
    pass


class LLMQuotaError(LLMError):
    """Rate limit or quota exhaustion (HTTP 429)."""
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

    def acquire(self) -> float:
        """Acquire 1 token, sleeping if necessary. Thread-safe."""
        with self._lock:
            now = self.time_fn()
            elapsed = now - self.last_refill
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
        }

        with self._accounting_lock:
            with open(calls_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

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
                    except Exception:
                        pass

        manifest_data = {
            "run_id": rid,
            "git_commit": get_git_commit(),
            "config_hash": get_config_hash(self.models_config_path),
            "providers_used": sorted(list(providers_used)),
            "models_used": sorted(list(models_used)),
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
                )
            except Exception:
                pass

        # Execute call with rate limiting and exponential backoff retry
        rate_limiter = self.rate_limiters.get(role)
        if not rate_limiter:
            rpm = cfg.get("rpm_limit", 15)
            rate_limiter = TokenBucketRateLimiter(
                rpm=rpm, time_fn=self.time_fn, sleep_fn=self.sleep_fn
            )
            self.rate_limiters[role] = rate_limiter

        attempts = 0
        max_retries = 5
        start_time = self.time_fn()
        raw_text = ""
        in_tokens = 0
        out_tokens = 0
        thinking_tokens = 0
        is_estimated = False

        while attempts < max_retries:
            attempts += 1
            rate_limiter.acquire()

            try:
                if provider == "mock":
                    raw_text, in_tokens, out_tokens, thinking_tokens, is_estimated = self._call_mock(
                        role, system, user, schema
                    )
                elif provider == "gemini":
                    raw_text, in_tokens, out_tokens, thinking_tokens, is_estimated = self._call_gemini(
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
                    raw_text, in_tokens, out_tokens, thinking_tokens, is_estimated = self._call_openai(
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

                # Successful network call
                break

            except (LLMAuthError, LLMBadRequestError) as e:
                # Fatal errors: never retry
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
                    attempts=attempts,
                    success=False,
                    estimated_tokens=False,
                )
                raise e

            except (LLMQuotaError, LLMError, urllib.error.URLError, TimeoutError) as e:
                if attempts >= max_retries:
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
                        attempts=attempts,
                        success=False,
                        estimated_tokens=False,
                    )
                    raise LLMError(f"Max retries ({max_retries}) exceeded for {role}: {e}")

                backoff = min(30.0, (2 ** (attempts - 1)) + random.uniform(0.1, 0.5))
                retry_after = getattr(e, "retry_after", None)
                sleep_duration = float(retry_after) if retry_after else backoff
                self.sleep_fn(sleep_duration)

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
                        repair_text, r_in, r_out, r_th, r_est = self._call_mock(
                            role, system, repair_prompt, schema
                        )
                    elif provider == "gemini":
                        repair_text, r_in, r_out, r_th, r_est = self._call_gemini(
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
                        repair_text, r_in, r_out, r_th, r_est = self._call_openai(
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
                        r_in, r_out, r_th, r_est = in_tokens, out_tokens, thinking_tokens, is_estimated

                    in_tokens += r_in
                    out_tokens += r_out
                    thinking_tokens += r_th
                    raw_text = repair_text
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
            res = self.call(**kwargs)
            return idx, res

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_worker, i, req) for i, req in enumerate(requests)
            ]
            for fut in concurrent.futures.as_completed(futures):
                idx, res = fut.result()
                results[idx] = res

        return [r for r in results if r is not None]

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
    ) -> Tuple[str, int, int, int, bool]:
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
            retry_after = e.headers.get("Retry-After")

            if e.code in (401, 403):
                raise LLMAuthError(f"Gemini Auth Error (HTTP {e.code}): {err_body}")
            elif e.code == 400:
                raise LLMBadRequestError(f"Gemini Client Error (HTTP 400): {err_body}")
            elif e.code == 429:
                err = LLMQuotaError(f"Gemini Quota/Rate Limit (HTTP 429): {err_body}")
                err.retry_after = retry_after
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

        return raw_text, in_tokens, total_out_tokens, thinking_tokens, is_estimated

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
    ) -> Tuple[str, int, int, int, bool]:
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
            retry_after = e.headers.get("Retry-After")

            if e.code in (401, 403):
                raise LLMAuthError(f"OpenAI Auth Error (HTTP {e.code}): {err_body}")
            elif e.code == 400:
                raise LLMBadRequestError(f"OpenAI Client Error (HTTP 400): {err_body}")
            elif e.code == 429:
                err = LLMQuotaError(f"OpenAI Quota/Rate Limit (HTTP 429): {err_body}")
                err.retry_after = retry_after
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

        return raw_text, in_tokens, out_tokens, reasoning_tokens, is_estimated

    def _call_mock(
        self,
        role: str,
        system: str,
        user: str,
        schema: Optional[Type[BaseModel]],
    ) -> Tuple[str, int, int, int, bool]:
        """Offline mock provider for testing."""
        with self._mock_lock:
            queue = self.mock_registry.get(role, [])
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
        return raw_text, in_tokens, out_tokens, 0, True


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
