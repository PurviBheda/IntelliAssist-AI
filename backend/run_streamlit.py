"""IntelliAssist AI - Streamlit Application Launcher.

Starts the Streamlit frontend configured to connect to the FastAPI backend.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

backend_dir = Path(__file__).resolve().parent
app_path = backend_dir / "streamlit_app" / "app.py"


def print_banner() -> None:
    """Display startup banner and instructions."""
    print("=" * 70)
    print("      IntelliAssist AI — Streamlit Frontend Launcher")
    print("=" * 70)
    print("Important Notice:")
    print("  The Streamlit application communicates with the FastAPI backend.")
    print("  Please ensure the backend is running in a separate terminal:")
    print()
    print("    cd \"d:\\IntelliAssist AI\\backend\"")
    print("    .\\.venv\\Scripts\\Activate.ps1")
    print("    python server.py")
    print()
    print("  Backend API:  http://127.0.0.1:8000")
    print("  Streamlit UI: http://localhost:8501")
    print("=" * 70)
    print()


def main() -> int:
    """Launch Streamlit using the active Python executable."""
    print_banner()

    if not app_path.is_file():
        print(f"[ERROR] Streamlit entrypoint not found at: {app_path}")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.headless=false",
    ]

    print(f"Launching Streamlit: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=str(backend_dir))
        return proc.returncode
    except KeyboardInterrupt:
        print("\nShutting down Streamlit application.")
        return 0
    except Exception as err:
        print(f"[ERROR] Failed to start Streamlit: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
