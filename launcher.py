"""
Cognify Launcher
-----------------
Entry point for the packaged .exe.

1. Checks if Ollama is reachable. If not, opens ollama.com.
2. Starts the Streamlit app in a subprocess.
3. Opens the browser at localhost:8501.
4. Waits for the user to close the terminal / window.
"""

import os
import sys
import time
import subprocess
import webbrowser

import requests

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
APP_PATH   = os.path.join(BASE_DIR, "src", "app.py")
PORT       = 8501
OLLAMA_URL = "http://localhost:11434"


def check_ollama() -> bool:
    try:
        r = requests.get(OLLAMA_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def main():
    print("=" * 50)
    print("  Cognify - Starting up")
    print("=" * 50)

    # ── 1. Check Ollama ──────────────────────────
    if not check_ollama():
        print("\n[!] Ollama is not running or not installed.")
        print("    Opening ollama.com so you can download it.")
        print("    After installing, run 'ollama serve' and restart Cognify.\n")
        webbrowser.open("https://ollama.com")
        input("Press Enter to exit...")
        sys.exit(1)

    print("[OK] Ollama is running.")

    # ── 2. Start Streamlit ───────────────────────
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "streamlit", "run", APP_PATH,
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
    ]

    print(f"[OK] Starting Cognify on http://localhost:{PORT} ...")
    proc = subprocess.Popen(cmd, cwd=BASE_DIR)

    # ── 3. Wait for server to be ready ───────────
    for _ in range(20):
        time.sleep(1)
        try:
            r = requests.get(f"http://localhost:{PORT}", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass

    # ── 4. Open browser ──────────────────────────
    webbrowser.open(f"http://localhost:{PORT}")
    print(f"[OK] Cognify is live at http://localhost:{PORT}")
    print("    Close this window to stop the app.\n")

    # ── 5. Keep alive until user closes ──────────
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down Cognify...")
        proc.terminate()


if __name__ == "__main__":
    main()
