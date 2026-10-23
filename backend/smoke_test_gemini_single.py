"""IntelliAssist AI - Fast Focused Single-Request Live Smoke Test for Gemini.

Performs exactly ONE lightweight API request against the configured Gemini model
to verify client initialization, connectivity, authentication, and generation latency.

Never logs, prints, or exposes API keys or sensitive credentials.

Usage:
    .\\backend\\.venv\\Scripts\\python.exe backend/smoke_test_gemini_single.py
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.rag_pipeline import (
    DEFAULT_GEMINI_MODEL,
    GeminiLLMClient,
    load_environment_variables,
)


def sanitize_message(msg: Any, secret: Optional[str] = None) -> str:
    """Ensure no secret tokens appear in output."""
    s = str(msg)
    if secret and secret in s:
        s = s.replace(secret, "[REDACTED_API_KEY]")
    s = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]", s)
    s = re.sub(r"(key=)[^&\s]+", r"\1[REDACTED_API_KEY]", s)
    return s


def classify_error(err_msg: str) -> str:
    """Classify error into actionable diagnostic category."""
    text = (err_msg or "").lower()
    if "429" in text or "rate limit" in text or "quota" in text or "resource_exhausted" in text:
        return "HTTP 429 - Quota Exceeded / Rate Limit Hit"
    if "503" in text or "500" in text or "overloaded" in text or "unavailable" in text:
        return "HTTP 503/500 - Model Service Overloaded / Temporarily Unavailable"
    if "timeout" in text or "timed out" in text:
        return "Timeout - API Request Exceeded Configured Timeout Boundary"
    if "empty or blocked" in text:
        return "Safety / Empty Response - Model Generation Output Was Empty"
    if "connection" in text or "network" in text:
        return "Network / Connection Error"
    return "Gemini API Generation Error"


def run_single_smoke_test() -> int:
    """Execute exactly ONE lightweight Gemini generation request."""
    print("=" * 70)
    print(" IntelliAssist AI — Single-Request Gemini Smoke Test ".center(70, "="))
    print("=" * 70)

    # 1. Load environment and verify key presence
    load_environment_variables()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    placeholder_keys = {
        "your_gemini_api_key_here",
        "your_actual_key_here",
        "your_api_key_here",
        "TODO",
        "CHANGEME",
    }
    if not api_key or not api_key.strip() or api_key.strip() in placeholder_keys:
        print("\n[SMOKE TEST STATUS: SKIPPED]")
        print("  GEMINI_API_KEY is not set or is set to a placeholder value.")
        print("  Please configure GEMINI_API_KEY in backend/.env to run live tests.")
        return 1

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    print(f"\nConfiguration:")
    print(f"  Target Model:   {model_name}")
    print(f"  Key Status:     Configured (Protected)")

    # 2. Execute single generation call
    print("\nSending single test request to Gemini Interactions API...")
    t0 = time.time()
    try:
        client = GeminiLLMClient(model_name=model_name)
        response = client.generate_response(
            system_instruction="You are a helpful assistant. Reply concisely.",
            prompt="Respond with exactly the single word: CONFIRMED",
        )
        latency = time.time() - t0
        print(f"\n[PASSED] Single Request Generation Succeeded!")
        print(f"  Latency:        {latency:.2f}s")
        print(f"  Model Output:   \"{response.strip()}\"")
        print("=" * 70)
        return 0

    except Exception as e:
        latency = time.time() - t0
        sanitized_err = sanitize_message(e, api_key)
        category = classify_error(sanitized_err)
        print(f"\n[FAILED] Single Request Generation Failed after {latency:.2f}s")
        print(f"  Category:       {category}")
        print(f"  Sanitized Error:{sanitized_err}")
        print("\nDiagnostic Advice:")
        if "429" in sanitized_err or "quota" in sanitized_err.lower():
            print("  * Gemini free-tier quota was temporarily exceeded. Wait 60 seconds before retrying.")
        elif "overloaded" in sanitized_err.lower() or "503" in sanitized_err:
            print("  * Google Gemini service is experiencing high load. The model API is temporarily overloaded.")
        elif "timeout" in sanitized_err.lower():
            print("  * The request timed out. Check GEMINI_TIMEOUT_SECONDS or your internet connection.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_single_smoke_test())
