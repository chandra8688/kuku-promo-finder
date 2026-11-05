"""Unit tests for Centralized Model Gateway (src/llm.py).

Tests caching, retry on 429, no-retry on 400, schema repair retry,
rate limiter token bucket, and accounting safety using mock provider.
All tests run 100% offline with zero network calls.
"""

import json
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel

from src.llm import (
    LLMBadRequestError,
    LLMQuotaError,
    LLMQuotaExhaustedError,
    LLMRepairFailedError,
    LLMResult,
    ModelGateway,
    TokenBucketRateLimiter,
    UsageLedger,
)

MOCK_ROLES_CONFIG = {
    "cheap": {
        "provider": "mock",
        "model_name": "mock-cheap",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "rpm_limit": 60,
    },
    "strong": {
        "provider": "mock",
        "model_name": "mock-strong",
        "temperature": 0.0,
        "max_output_tokens": 2048,
        "rpm_limit": 60,
    },
    "judge": {
        "provider": "mock",
        "model_name": "mock-judge",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "rpm_limit": 60,
    },
}


class SampleScore(BaseModel):
    score: int
    label: str


def test_caching_hit_and_miss(tmp_path: Path):
    """Verify caching produces hit on identical input and bypasses when disabled."""
    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=True,
    )
    # Register mock responses
    gw.register_mock_response("cheap", "Response Alpha")
    gw.register_mock_response("cheap", "Response Beta")

    # Call 1: Cache Miss
    res1 = gw.call(role="cheap", system="sys", user="user 1", stage="test", prompt_version="v1")
    assert res1.text == "Response Alpha"
    assert res1.cached is False
    assert res1.attempts == 1

    # Call 2: Cache Hit (exact same input)
    res2 = gw.call(role="cheap", system="sys", user="user 1", stage="test", prompt_version="v1")
    assert res2.text == "Response Alpha"
    assert res2.cached is True
    assert res2.attempts == 1

    # Call 3: Cache Bypass with enable_cache=False
    res3 = gw.call(
        role="cheap",
        system="sys",
        user="user 1",
        stage="test",
        prompt_version="v1",
        enable_cache=False,
    )
    assert res3.text == "Response Beta"
    assert res3.cached is False


def test_retry_on_simulated_429(tmp_path: Path):
    """Verify exponential backoff retry on HTTP 429 rate limit."""
    sleeps: List[float] = []

    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
        sleep_fn=lambda s: sleeps.append(s),
    )

    # Queue 429 error followed by successful response
    gw.register_mock_response("cheap", LLMQuotaError("Simulated 429"))
    gw.register_mock_response("cheap", "Recovered Text")

    res = gw.call(role="cheap", system="sys", user="retry test", stage="test")
    assert res.text == "Recovered Text"
    assert res.attempts == 2
    assert len(sleeps) == 1
    assert sleeps[0] > 0.0


def test_no_retry_on_simulated_400(tmp_path: Path):
    """Verify client errors (HTTP 400) fail immediately with no retry."""
    sleeps: List[float] = []

    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
        sleep_fn=lambda s: sleeps.append(s),
    )

    # Queue 400 error
    gw.register_mock_response("cheap", LLMBadRequestError("Simulated 400 Bad Request"))

    with pytest.raises(LLMBadRequestError):
        gw.call(role="cheap", system="sys", user="bad request test", stage="test")

    # Ensure no retry sleeps took place
    assert len(sleeps) == 0


def test_schema_repair_retry_succeeds(tmp_path: Path):
    """Verify schema validation error triggers exactly one repair retry and succeeds."""
    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
    )

    # First attempt: invalid JSON
    gw.register_mock_response("cheap", "Not valid json at all")
    # Repair attempt: valid JSON conforming to SampleScore
    gw.register_mock_response("cheap", '{"score": 9, "label": "Excellent"}')

    res = gw.call(
        role="cheap",
        system="sys",
        user="score please",
        schema=SampleScore,
        stage="test",
    )
    assert res.parsed is not None
    assert isinstance(res.parsed, SampleScore)
    assert res.parsed.score == 9
    assert res.parsed.label == "Excellent"
    assert res.attempts == 2


def test_schema_repair_fails_cleanly(tmp_path: Path):
    """Verify schema validation error fails cleanly if repair attempt also fails."""
    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
    )

    # Both attempts invalid
    gw.register_mock_response("cheap", "Invalid 1")
    gw.register_mock_response("cheap", '{"wrong_field": "still invalid"}')

    with pytest.raises(LLMRepairFailedError):
        gw.call(
            role="cheap",
            system="sys",
            user="score please",
            schema=SampleScore,
            stage="test",
        )


def test_rate_limiter_spacing():
    """Verify token bucket rate limiter spaces calls using fake clock."""
    current_time = [100.0]
    sleeps: List[float] = []

    def fake_now() -> float:
        return current_time[0]

    def fake_sleep(dur: float) -> None:
        sleeps.append(dur)
        current_time[0] += dur

    # 60 RPM = 1 token/sec, capacity 60
    limiter = TokenBucketRateLimiter(rpm=60, time_fn=fake_now, sleep_fn=fake_sleep)

    # Consume all 60 initial capacity tokens
    for _ in range(60):
        w = limiter.acquire()
        assert w == 0.0

    # 61st token must trigger wait
    w = limiter.acquire()
    assert w > 0.0
    assert len(sleeps) == 1
    assert round(sleeps[0], 2) == 1.00


def test_accounting_file_has_no_screenplay_text(tmp_path: Path):
    """Verify calls.jsonl and manifest.json contain only telemetry, never prompt/response text."""
    runs_dir = tmp_path / "runs"
    run_id = "test_run_001"

    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(runs_dir),
        enable_cache=False,
    )

    secret_prompt = "TOP_SECRET_PROMPT_CONTENT"
    secret_response = "TOP_SECRET_RESPONSE_CONTENT"
    gw.register_mock_response("cheap", secret_response)

    gw.call(
        role="cheap",
        system="secret system",
        user=secret_prompt,
        stage="test_stage",
        run_id=run_id,
    )

    calls_file = runs_dir / run_id / "calls.jsonl"
    assert calls_file.exists()

    with open(calls_file, "r", encoding="utf-8") as f:
        content = f.read()
        lines = [json.loads(l) for l in content.splitlines() if l.strip()]

    # Verify secret text is NOT logged
    assert secret_prompt not in content
    assert secret_response not in content

    # Verify accounting fields exist
    entry = lines[0]
    assert entry["stage"] == "test_stage"
    assert entry["role"] == "cheap"
    assert entry["provider"] == "mock"
    assert "tokens_in" in entry
    assert "tokens_out" in entry
    assert "latency_s" in entry
    assert "success" in entry

    # Verify manifest
    manifest_path = gw.write_manifest(run_id=run_id)
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_calls"] == 1
    assert "safety_warning" in manifest
    assert "git_commit" in manifest


def test_call_many_concurrency(tmp_path: Path):
    """Verify call_many executes requests concurrently and returns results in matching order."""
    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
    )

    gw.register_mock_response("cheap", "Response 1")
    gw.register_mock_response("cheap", "Response 2")
    gw.register_mock_response("cheap", "Response 3")
    gw.register_mock_response("cheap", "Response 4")

    requests = [
        {"role": "cheap", "system": "s", "user": f"u {i}", "stage": "test"}
        for i in range(1, 5)
    ]

    results = gw.call_many(requests, max_workers=2)
    assert len(results) == 4
    texts = [r.text for r in results]
    assert texts == ["Response 1", "Response 2", "Response 3", "Response 4"]


def test_usage_ledger_rolling_24h(tmp_path: Path):
    """Verify UsageLedger persists and counts calls over a rolling 24h window."""
    ledger_file = tmp_path / "usage.json"
    ledger = UsageLedger(ledger_path=ledger_file, runs_dir=tmp_path / "runs")

    ledger.record_call(model="test-model", success=True, tokens_in=100, tokens_out=50)
    ledger.record_call(model="test-model", success=False, tokens_in=50, tokens_out=0)

    stats = ledger.get_stats_24h("test-model")
    assert stats["successful_calls"] == 1
    assert stats["failed_calls"] == 1
    assert stats["total_calls"] == 2

    # Check remaining daily quota
    rem = ledger.get_remaining_daily_quota("test-model", rpd_limit=20)
    assert rem == 18

    # Limit exceeded check
    rem_small = ledger.get_remaining_daily_quota("test-model", rpd_limit=2)
    assert rem_small == 0


def test_quota_exhausted_no_retry(tmp_path: Path):
    """Verify LLMQuotaExhaustedError is never retried and raised immediately."""
    sleeps = []
    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
        sleep_fn=lambda s: sleeps.append(s),
    )

    # Register quota exhausted error
    gw.register_mock_response("cheap", LLMQuotaExhaustedError("Daily quota exhausted"))

    with pytest.raises(LLMQuotaExhaustedError):
        gw.call(role="cheap", system="sys", user="user", stage="test")

    # Must NOT have retried or slept
    assert len(sleeps) == 0


def test_check_stage_quota_refusal_and_force(tmp_path: Path):
    """Verify gateway refuses to start when required calls exceed quota, unless forced."""
    roles = {
        "cheap": {
            "provider": "mock",
            "model_name": "mock-cheap",
            "rpm_limit": 60,
            "rpd_limit": 5,
        }
    }
    gw = ModelGateway(
        roles_config=roles,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
    )

    # 4 calls already used in ledger
    for _ in range(4):
        gw.usage_ledger.record_call("mock-cheap", success=True)

    rem = gw.get_remaining_quota("cheap")
    assert rem == 1

    # Asking for 3 calls with only 1 remaining must raise without force
    with pytest.raises(LLMQuotaExhaustedError, match="Refusing to start"):
        gw.check_stage_quota("cheap", required_calls=3, force=False)

    # With force=True, must pass without raising
    gw.check_stage_quota("cheap", required_calls=3, force=True)

