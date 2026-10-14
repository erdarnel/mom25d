"""Open-stub notch filter (openEMS MSL_NotchFilter tutorial geometry),
compared with the openEMS FDTD reference stored in Engine/tests/data."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")
REFDATA = os.path.join(here, "..", "..", "..", "Engine", "tests", "data")

from mom25d import GHz, Solver, mm  # noqa: E402
from notch_filter import layout, substrate  # noqa: E402

mesh = layout().mesh(max_cell=0.5 * mm, edge_cell=0.05 * mm)

sol = Solver(mesh, substrate())
f = np.unique(np.r_[np.linspace(0.5, 7, 27), np.linspace(3.5, 3.85, 8)]) * GHz
res = sol.sweep(f)
S50 = res.renormalize(50)
res.write_touchstone(os.path.join(RESULTS, "notch_filter.s2p"), zref=50)

ref = np.loadtxt(os.path.join(REFDATA, "notch_openems.txt"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[2, 1]))
ax[0].plot(ref[:, 0] / GHz, 20 * np.log10(ref[:, 2]), "k-", lw=1, label="S21 openEMS (FDTD)")
ax[0].plot(ref[:, 0] / GHz, 20 * np.log10(ref[:, 1]), "k--", lw=1, label="S11 openEMS (FDTD)")
ax[0].plot(f / GHz, res.db(1, 0, S50), "o", ms=4, color="C3", label="S21 mom25d")
ax[0].plot(f / GHz, res.db(0, 0, S50), "s", ms=4, color="C0", label="S11 mom25d")
ax[0].set(xlabel="frequency [GHz]", ylabel="dB", ylim=(-45, 2), title="Open-stub notch filter, 50 Ω")
ax[0].grid(alpha=0.3)
ax[0].legend()
mesh.plot(ax[1])
ax[1].set_title(f"mesh: {mesh.n_basis} RWG unknowns")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "notch_filter.png"), dpi=130)
print("saved notch_filter.png / notch_filter.s2p")
