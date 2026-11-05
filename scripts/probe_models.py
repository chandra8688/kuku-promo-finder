"""Capability Probe for Candidate Models (scripts/probe_models.py).

Tests:
1. system instruction accepted
2. JSON schema output validates against a small Pydantic schema
3. thinkingBudget accepted or rejected (records error type)
4. latency of a trivial call
Empirical measurements only, dummy prompts only, zero screenplay text.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv(".env")

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemma-4-31b-it",
]


class ProbeOutput(BaseModel):
    status: str
    number: int


def send_probe_request(
    model: str,
    api_key: str,
    system: Optional[str] = None,
    user: str = "Output JSON with status 'ok' and number 42.",
    use_json_mime: bool = False,
    thinking_budget: Optional[int] = None,
) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    gen_config: Dict[str, Any] = {
        "temperature": 0.0,
        "maxOutputTokens": 200,
    }
    if use_json_mime:
        gen_config["responseMimeType"] = "application/json"
    if thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

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
        headers={"Content-Type": "application/json", "User-Agent": "promo-finder/probe"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = round(time.time() - t0, 3)
            return {"success": True, "data": data, "latency_s": lat, "error": None}
    except urllib.error.HTTPError as e:
        lat = round(time.time() - t0, 3)
        err_body = e.read().decode("utf-8", errors="replace")
        return {"success": False, "http_code": e.code, "error": err_body, "latency_s": lat}
    except Exception as e:
        lat = round(time.time() - t0, 3)
        return {"success": False, "http_code": None, "error": str(e), "latency_s": lat}


def run_probe():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in .env")
        return

    results = []
    total_calls = 0

    print("=" * 80)
    print("PROBING 5 CANDIDATE MODELS FOR PROMO-FINDER ROLES")
    print("=" * 80)

    for m in MODELS:
        print(f"\n[Probing {m}]...")
        probe_res = {
            "model": m,
            "system_instruction": "UNTRIED",
            "json_schema": "UNTRIED",
            "thinking_budget": "UNTRIED",
            "latency_s": "n/a",
            "notes": "",
            "calls_used": 0,
            "quota_impact": "None",
        }

        # Test 1: Full combo (System instruction + JSON mime + thinkingBudget=0)
        total_calls += 1
        probe_res["calls_used"] += 1
        r1 = send_probe_request(
            model=m,
            api_key=api_key,
            system="You are a strict JSON data generator.",
            user="Output a JSON object with keys 'status' ('ok') and 'number' (42).",
            use_json_mime=True,
            thinking_budget=0,
        )

        if r1["success"]:
            probe_res["system_instruction"] = "PASS"
            probe_res["thinking_budget"] = "PASS (accepted 0)"
            probe_res["latency_s"] = str(r1["latency_s"])
            probe_res["quota_impact"] = f"1 call on {m} daily quota"

            # Check JSON
            raw_text = ""
            cands = r1["data"].get("candidates", [])
            if cands and "content" in cands[0]:
                parts = cands[0]["content"].get("parts", [])
                raw_text = "".join(p.get("text", "") for p in parts).strip()
            try:
                parsed = ProbeOutput.model_validate_json(raw_text)
                probe_res["json_schema"] = f"PASS ({parsed.status}, {parsed.number})"
            except Exception as e:
                probe_res["json_schema"] = f"FAIL (validation: {e})"
        else:
            # Analyze error
            err_str = str(r1.get("error", ""))
            http_code = r1.get("http_code")
            probe_res["latency_s"] = str(r1["latency_s"])

            if http_code == 429:
                probe_res["notes"] = f"HTTP 429 Quota Exhausted / Rate Limited: {err_str[:120]}"
                probe_res["quota_impact"] = "Hit 429"
                results.append(probe_res)
                continue

            # If thinkingBudget caused the error, test without thinkingBudget
            if "thinking" in err_str.lower() or "budget" in err_str.lower() or http_code == 400:
                probe_res["thinking_budget"] = f"REJECTED ({http_code}): {err_str[:80]}"
                total_calls += 1
                probe_res["calls_used"] += 1
                r2 = send_probe_request(
                    model=m,
                    api_key=api_key,
                    system="You are a strict JSON data generator.",
                    user="Output a JSON object with keys 'status' ('ok') and 'number' (42).",
                    use_json_mime=True,
                    thinking_budget=None,
                )
                if r2["success"]:
                    probe_res["system_instruction"] = "PASS"
                    probe_res["latency_s"] = str(r2["latency_s"])
                    probe_res["quota_impact"] = f"2 calls on {m} daily quota"
                    raw_text = ""
                    cands = r2["data"].get("candidates", [])
                    if cands and "content" in cands[0]:
                        parts = cands[0]["content"].get("parts", [])
                        raw_text = "".join(p.get("text", "") for p in parts).strip()
                    try:
                        parsed = ProbeOutput.model_validate_json(raw_text)
                        probe_res["json_schema"] = f"PASS ({parsed.status}, {parsed.number})"
                    except Exception as e:
                        probe_res["json_schema"] = f"FAIL (validation: {e})"
                else:
                    err2 = str(r2.get("error", ""))
                    probe_res["notes"] = f"Call 2 failed ({r2.get('http_code')}): {err2[:120]}"
                    if "system" in err2.lower():
                        probe_res["system_instruction"] = "REJECTED"
                    if "json" in err2.lower() or "mime" in err2.lower():
                        probe_res["json_schema"] = "REJECTED"
            else:
                probe_res["notes"] = f"HTTP {http_code}: {err_str[:120]}"

        results.append(probe_res)

    print("\n" + "=" * 95)
    print("CAPABILITY PROBE SUMMARY TABLE")
    print("=" * 95)
    hdr = f"{'Model':<22} | {'System Prompt':<14} | {'JSON Schema':<22} | {'ThinkingBudget':<20} | {'Latency':<8} | {'Calls'}"
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        print(
            f"{res['model']:<22} | "
            f"{res['system_instruction']:<14} | "
            f"{res['json_schema']:<22} | "
            f"{res['thinking_budget']:<20} | "
            f"{res['latency_s']+'s':<8} | "
            f"{res['calls_used']} ({res['quota_impact']})"
        )
        if res["notes"]:
            print(f"   Note: {res['notes']}")
    print("-" * len(hdr))
    print(f"Total API Calls Made: {total_calls} (limit was 12 calls)")


if __name__ == "__main__":
    run_probe()
