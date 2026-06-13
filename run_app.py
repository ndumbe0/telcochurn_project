import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    app_path = Path(__file__).resolve().parent / "src" / "app" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    main()
