#!/usr/bin/env python3
"""List available models from configured providers.

Queries provider APIs using credentials from .env and prints available model names
and supported generation methods.

NOTE ON RATE LIMITS:
The default rate limit in promo-finder is 15 RPM (Requests Per Minute). This is an
ASSUMPTION for standard free-tier accounts. You should verify your actual rate limit
and quota at https://ai.google.dev or your provider console.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
import yaml


def list_gemini_models(api_key: str) -> None:
    print("\n--- Google Gemini Models ---")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "promo-finder/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            print(f"Total models returned: {len(models)}")
            for m in models:
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    name = m.get("name", "").replace("models/", "")
                    print(f"  - {name} (supported: {', '.join(methods)})")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
    except Exception as e:
        print(f"Failed to query Gemini models: {e}")


def list_openai_models(api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
    print(f"\n--- OpenAI-Compatible Models ({base_url}) ---")
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "promo-finder/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            print(f"Total models returned: {len(models)}")
            for m in models:
                m_id = m.get("id", "")
                print(f"  - {m_id}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
    except Exception as e:
        print(f"Failed to query OpenAI-compatible models: {e}")


def main() -> None:
    load_dotenv(".env")

    print("=================================================================")
    print("PROMO-FINDER: Provider Model Inspection")
    print("=================================================================")
    print("REMINDER: The 15 RPM default limit is an ASSUMPTION.")
    print("Verify your actual account rate limits and daily quota at:")
    print("  https://ai.google.dev")
    print("=================================================================")

    # Load configured providers from models.yaml
    models_path = Path("config/models.yaml")
    if not models_path.exists():
        print("config/models.yaml not found.")
        sys.exit(1)

    with open(models_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    roles = cfg.get("roles", {})
    queried_providers = set()

    for role_name, rcfg in roles.items():
        provider = rcfg.get("provider", "").lower()
        key_env = rcfg.get("api_key_env", "GEMINI_API_KEY")
        base_url = rcfg.get("base_url")

        provider_tag = f"{provider}:{base_url}"
        if provider_tag in queried_providers:
            continue
        queried_providers.add(provider_tag)

        api_key = os.environ.get(key_env, "").strip()
        if not api_key:
            print(
                f"\n[!] Missing key for {role_name} (provider: {provider}): "
                f"Environment variable '{key_env}' is not set in .env."
            )
            continue

        if provider == "gemini":
            list_gemini_models(api_key)
        elif provider == "openai_compatible":
            url = base_url or "https://api.openai.com/v1"
            list_openai_models(api_key, base_url=url)
        elif provider == "mock":
            print(f"\n--- Mock Provider ({role_name}) ---")
            print("  - mock-instant (offline test provider)")
        else:
            print(f"\n[!] Unknown provider type: {provider}")


if __name__ == "__main__":
    main()

