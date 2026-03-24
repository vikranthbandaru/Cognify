"""
Run this script once to build Cognify.exe using PyInstaller.

    python build_exe.py

Output: dist/Cognify/Cognify.exe
"""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "Cognify",
        "--onedir",                      # folder bundle (faster startup than --onefile)
        "--noconfirm",                   # overwrite previous build without asking
        "--clean",
        "--add-data", f"src{os.pathsep}src",           # bundle entire src/ folder
        "--add-data", f".env{os.pathsep}.",            # bundle .env
        "--add-data", f".streamlit{os.pathsep}.streamlit",
        "--hidden-import", "streamlit",
        "--hidden-import", "litellm",
        "--hidden-import", "fitz",
        "--hidden-import", "trafilatura",
        "--hidden-import", "altair",
        "--hidden-import", "pandas",
        "--hidden-import", "matplotlib",
        "--collect-all", "streamlit",
        "--collect-all", "altair",
        "--collect-all", "litellm",
        "--collect-all", "tiktoken",
        "--copy-metadata", "tiktoken",
        "--hidden-import", "tiktoken_ext.openai_public",
        "--hidden-import", "tiktoken_ext.bpe",
        "launcher.py",
    ]

    print("Building Cognify.exe ...")
    print("This takes 3-5 minutes on first run.\n")
    result = subprocess.run(cmd, cwd=HERE)

    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("  Build complete!")
        print(f"  Executable: {os.path.join(HERE, 'dist', 'Cognify', 'Cognify.exe')}")
        print("=" * 50)
        print("\nTo distribute:")
        print("  Zip the entire dist/Cognify/ folder and share it.")
        print("  Users must have Ollama installed (ollama.com).")
        print("  First run: launch Cognify.exe, it opens the browser automatically.")
    else:
        print("\nBuild failed. Check the output above for errors.")


if __name__ == "__main__":
    main()
