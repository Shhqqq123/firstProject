from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def run_web() -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"])


def run_streamlit_server(port: str) -> int:
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).resolve().parent / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode",
        "false",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--browser.serverAddress",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
    ]
    return streamlit_cli.main()


if __name__ == "__main__":
    if "--streamlit-server" in sys.argv:
        idx = sys.argv.index("--streamlit-server")
        port = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "8501"
        raise SystemExit(run_streamlit_server(port))
    if "--web" in sys.argv:
        raise SystemExit(run_web())
    try:
        from desktop_app import main as desktop_main
    except Exception:
        raise SystemExit(run_web())
    raise SystemExit(desktop_main())
