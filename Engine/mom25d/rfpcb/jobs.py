"""Submitting EM runs to a simulation manager (simmgr: `simmgr.py submit ...`), with a 3-D model page (--model3d) and
an optional pre-flight check of memory and other simulations.

    RFPCB_SIMMGR     path of simmgr.py (default ~/simmgr/simmgr.py)
    RFPCB_PREFLIGHT  optional shell script printing free memory / running simulations before a submit"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

SIMMGR = Path(os.environ.get("RFPCB_SIMMGR", Path.home() / "simmgr" / "simmgr.py"))
PREFLIGHT = Path(os.environ["RFPCB_PREFLIGHT"]) if os.environ.get("RFPCB_PREFLIGHT") else None


def preflight():
    if PREFLIGHT is not None and PREFLIGHT.exists():
        return subprocess.run(["bash", str(PREFLIGHT)], capture_output=True, text=True).stdout
    return "(no pre-flight script)"


def dense_gb(n_unknowns):
    """Dense solve memory: about 48 N^2 bytes (three complex N x N matrices)."""
    return 48.0 * n_unknowns ** 2 / 1e9


def submit(structure, variation, command, cwd, need_gb, est_min, model3d, solver="mom25d", dry=False):
    """Queue a run in simmgr; MOM25D_PATH and PYTHONUNBUFFERED are passed through. Returns simmgr's output."""
    if not SIMMGR.exists():
        raise RuntimeError(f"simmgr not found at {SIMMGR}")
    args = [sys.executable, str(SIMMGR), "submit", "--structure", structure, "--variation", variation,
            "--need-gb", f"{need_gb:g}", "--solver", solver, "--cwd", str(cwd), "--est-min", f"{est_min:g}",
            "--model3d", str(model3d), "--env", f"MOM25D_PATH={os.environ.get('MOM25D_PATH', '')}",
            "--env", f"PYTHONPATH={Path(__file__).resolve().parents[2]}",   # Engine/, so `-m mom25d.rfpcb` imports
            "--env", "PYTHONUNBUFFERED=1", "--"] + list(command)
    if dry:
        return " ".join(shlex.quote(a) for a in args)
    return subprocess.run(args, capture_output=True, text=True).stdout
