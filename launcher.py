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
    import threading
    from streamlit.web.cli import main as st_main

    def open_browser():
        # Wait for the Streamlit server to be ready
        for _ in range(20):
            time.sleep(1)
            try:
                r = requests.get(f"http://localhost:{PORT}", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
        
        webbrowser.open(f"http://localhost:{PORT}")
        print(f"[OK] Cognify is live at http://localhost:{PORT}")
        print("    Close this window to stop the app.\n")

    # Start the thread to open the browser
    threading.Thread(target=open_browser, daemon=True).start()

    print(f"[OK] Starting Cognify on http://localhost:{PORT} ...")
    
    # Run Streamlit natively in the same process
    # This prevents the infinite PyInstaller self-spawn loop
    sys.argv = [
        "streamlit", "run", APP_PATH,
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
    ]
    
    try:
        sys.exit(st_main())
    except Exception as e:
        print(f"\n[!] Error running Streamlit: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
