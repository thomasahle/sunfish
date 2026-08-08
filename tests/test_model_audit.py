"""CI hook for the model-code drift guard (formal/scripts/model_audit.py).

Fails whenever an audited region of sunfish.py (the regions the Lean
model in formal/ was audited against -- see formal/README.md's 'Model
fidelity' section) changes without a same-commit re-audit + hash
refresh.  This makes silent code/model divergence impossible until the
leanpy track checks the correspondence itself.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_audited_regions_match_model_audit():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "formal" / "scripts" / "model_audit.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "model-code drift detected:\n" + proc.stdout + proc.stderr
    )
