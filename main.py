from __future__ import annotations

import subprocess
import sys


def run_web() -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    if "--web" in sys.argv:
        raise SystemExit(run_web())
    try:
        from desktop_app import main as desktop_main
    except Exception:
        raise SystemExit(run_web())
    raise SystemExit(desktop_main())
