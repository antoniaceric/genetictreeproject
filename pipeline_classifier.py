import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent

# Scripts to run
SCRIPTS = [
    "classifier/matchIDs_case_control.py",   # creates ID_LifeAndBrain_to_Group.csv
    "classifier/case_control_labels.py",     # creates case_control_labels.csv
    "classifier/xgb_classifier.py",          # trains classifier, outputs scores and plots
]

def run_script(script_name: str):
    script_path = SCRIPT_DIR / script_name
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run([PYTHON, str(script_path)])
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

if __name__ == "__main__":
    print("Starting classifier pipeline...\n")
    for script in SCRIPTS:
        run_script(script)
    print("\nAll steps completed successfully.")
