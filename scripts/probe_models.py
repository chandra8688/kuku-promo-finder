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

PROBE_MODELS = [
    "gemini-3.8-flash",
    "gemma-4-31b-it",
]


class ProbeOutput(BaseModel):
    status: str
    number: int


def send_probe_request(
    model: str,
    api_key: str,
    system: Optional[str] = None,
    user: str = "Reply with the single word: OK.",
    use_json_mime: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    clean_model = model.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"

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
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = round(time.time() - t0, 3)
            return {"success": True, "data": data, "latency_s": lat, "error": None}
    except urllib.error.HTTPError as e:
        lat = round(time.time() - t0, 3)
        err_body = e.read().decode("utf-8", errors="replace")
        return {"success": False, "http_code": e.code, "error": err_body, "latency_s": lat}
    except (urllib.error.URLError, TimeoutError) as e:
        lat = round(time.time() - t0, 3)
        return {"success": False, "http_code": "TIMEOUT", "error": str(e), "latency_s": lat}
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

    print("=" * 85)
    print("RE-PROBING CANDIDATE MODELS (gemini-3.8-flash & gemma-4-31b-it)")
    print("=" * 85)

    for m in PROBE_MODELS:
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

        # 1. Probe System Instruction
        total_calls += 1
        probe_res["calls_used"] += 1
        print("  - Call 1: Testing systemInstruction...")
        r_sys = send_probe_request(
            model=m,
            api_key=api_key,
            system="You are a strict test assistant. Follow instructions precisely.",
            user="Reply with the single word: OK.",
            use_json_mime=False,
            thinking_budget=None,
            timeout_s=120,
        )
        if r_sys["success"]:
            probe_res["system_instruction"] = "PASS"
            probe_res["latency_s"] = str(r_sys["latency_s"])
        elif r_sys.get("http_code") == "TIMEOUT":
            probe_res["system_instruction"] = "TIMEOUT"
            probe_res["latency_s"] = f">120s"
        else:
            err = r_sys.get("error", "")
            probe_res["system_instruction"] = f"FAIL (HTTP {r_sys.get('http_code')})"
            probe_res["notes"] += f"sys_err: {err[:80]}; "

        # 2. Probe JSON Schema Output
        total_calls += 1
        probe_res["calls_used"] += 1
        print("  - Call 2: Testing JSON schema output...")
        use_sys = "You are a JSON generator." if probe_res["system_instruction"] == "PASS" else None
        user_prompt = "Output JSON strictly with keys 'status' ('ok') and 'number' (42)."
        if not use_sys:
            user_prompt = "You are a JSON generator. " + user_prompt

        r_json = send_probe_request(
            model=m,
            api_key=api_key,
            system=use_sys,
            user=user_prompt,
            use_json_mime=True,
            thinking_budget=None,
            timeout_s=120,
        )
        if r_json["success"]:
            raw_text = ""
            cands = r_json["data"].get("candidates", [])
            if cands and "content" in cands[0]:
                parts = cands[0]["content"].get("parts", [])
                raw_text = "".join(p.get("text", "") for p in parts).strip()
            try:
                parsed = ProbeOutput.model_validate_json(raw_text)
                probe_res["json_schema"] = f"PASS ({parsed.status}, {parsed.number})"
            except Exception as e:
                probe_res["json_schema"] = f"FAIL (validation: {e})"
        elif r_json.get("http_code") == "TIMEOUT":
            probe_res["json_schema"] = "TIMEOUT"
        else:
            err = r_json.get("error", "")
            probe_res["json_schema"] = f"FAIL (HTTP {r_json.get('http_code')})"
            probe_res["notes"] += f"json_err: {err[:80]}; "

        # 3. Probe thinkingBudget
        total_calls += 1
        probe_res["calls_used"] += 1
        print("  - Call 3: Testing thinkingBudget...")
        r_th = send_probe_request(
            model=m,
            api_key=api_key,
            system=None,
            user="Reply with the single word: OK.",
            use_json_mime=False,
            thinking_budget=0,
            timeout_s=120,
        )
        if r_th["success"]:
            probe_res["thinking_budget"] = "PASS (accepted 0)"
        elif r_th.get("http_code") == "TIMEOUT":
            probe_res["thinking_budget"] = "TIMEOUT"
        else:
            err = r_th.get("error", "")
            probe_res["thinking_budget"] = f"FAIL (HTTP {r_th.get('http_code')})"
            probe_res["notes"] += f"tb_err: {err[:80]}; "

        probe_res["quota_impact"] = f"{probe_res['calls_used']} calls"
        results.append(probe_res)

    print("\n" + "=" * 95)
    print("RE-PROBE CAPABILITY RESULTS TABLE")
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
            f"{res['latency_s']+'s' if not res['latency_s'].endswith('s') else res['latency_s']:<8} | "
            f"{res['calls_used']} ({res['quota_impact']})"
        )
        if res["notes"]:
            print(f"   Note: {res['notes']}")
    print("-" * len(hdr))
    print(f"Total API Calls Made: {total_calls} (limit was 8 calls)")


if __name__ == "__main__":
    run_probe()


