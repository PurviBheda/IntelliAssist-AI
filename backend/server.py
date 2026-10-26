"""IntelliAssist AI - Server Entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import uvicorn

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.rag_pipeline import load_environment_variables

if __name__ == "__main__":
    load_environment_variables()
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting IntelliAssist AI backend server on http://{host}:{port}")
    uvicorn.run("app.server:app", host=host, port=port, reload=False)
