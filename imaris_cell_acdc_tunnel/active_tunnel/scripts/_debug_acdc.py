"""Quick diagnostic: call ACDC run_cli directly and print any errors."""
import sys
import traceback

ini_path = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:\Users\wasee\Desktop\imaris_cellacdc_tunnel - fullTry - Version2 - Copy"
    r"\optuna_trials\thresholding_phase1\p1_trial_0000\trial.ini"
)

print(f"[diag] Testing ACDC CLI with: {ini_path}")

try:
    from cellacdc import _run
    print("[diag] cellacdc imported OK")
except Exception as e:
    print(f"[diag] IMPORT FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    _run.run_cli(ini_path)
    print("[diag] run_cli completed OK")
except Exception as e:
    print(f"[diag] run_cli FAILED: {e}")
    traceback.print_exc()
    sys.exit(2)
