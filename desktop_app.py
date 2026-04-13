from __future__ import annotations

import argparse
import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "app.py"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _build_streamlit_cmd(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
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


def _start_streamlit(port: int, verbose: bool) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and not verbose else 0
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    proc = subprocess.Popen(
        _build_streamlit_cmd(port),
        cwd=str(BASE_DIR),
        env=env,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
    )
    return proc


def _wait_for_streamlit(port: int, timeout_sec: int = 45) -> bool:
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    end = time.time() + timeout_sec
    while time.time() < end:
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.3)
    return False


def _stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_desktop(debug: bool = False, port: int | None = None) -> int:
    try:
        import webview
    except ImportError:
        print("缺少依赖 pywebview，请先执行：pip install pywebview", file=sys.stderr)
        return 2

    use_port = port or _find_free_port()
    proc = _start_streamlit(use_port, verbose=debug)
    atexit.register(_stop_process, proc)
    ok = _wait_for_streamlit(use_port)
    if not ok:
        _stop_process(proc)
        print("内置 Streamlit 服务启动失败。", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{use_port}"
    window = webview.create_window(
        title="乳腺风险评估系统",
        url=url,
        width=1480,
        height=920,
        min_size=(1200, 760),
    )

    def _closed() -> None:
        _stop_process(proc)

    window.events.closed += _closed
    try:
        webview.start(debug=debug)
    finally:
        _stop_process(proc)
    return 0


def run_smoke_test(port: int | None = None) -> int:
    use_port = port or _find_free_port()
    proc = _start_streamlit(use_port, verbose=True)
    ok = _wait_for_streamlit(use_port, timeout_sec=30)
    _stop_process(proc)
    if not ok:
        print("冒烟测试失败：服务健康检查未通过。")
        return 1
    print(f"冒烟测试通过：Streamlit 健康检查端口可用（{use_port}）。")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="乳腺风险评估系统桌面启动器")
    parser.add_argument("--debug", action="store_true", help="显示 Streamlit 日志并启用 webview 调试模式。")
    parser.add_argument("--port", type=int, default=None, help="指定内置服务端口（可选）。")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="不打开桌面窗口，仅启动服务并执行健康检查。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return run_smoke_test(port=args.port)
    return run_desktop(debug=args.debug, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
