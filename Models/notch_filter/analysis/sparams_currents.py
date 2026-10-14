"""Current distribution with S-parameter blocks connected between ports.

The open-stub notch filter (examples/current_density.py) is solved once;
port 2 is then terminated by three different S-parameter "blocks" (1-port
Touchstone files: matched 50 ohm, open, short) with `SweepResult.connect`.
`Network.currents` recombines the layout's port solutions with the waves
each block sends back, so the |J| maps show the standing waves the
termination sets up -- no new EM solve per block. Writes
results/sparams_currents.png.
"""

import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")
REFDATA = os.path.join(here, "..", "..", "..", "Engine", "tests", "data")

from mom25d import GHz, Solver, mm  # noqa: E402
from notch_filter import layout, substrate  # noqa: E402

mesh = layout().mesh(max_cell=0.25 * mm, edge_cell=0.04 * mm)
sol = Solver(mesh, substrate(), verbose=False)
f = np.array([2.0]) * GHz
res = sol.sweep(f, keep_currents=True, interp=None)

tmp = tempfile.mkdtemp()
blocks = {}
for name, s in (("matched 50 ohm", 0.0), ("open", 1.0), ("short", -1.0)):
    p = os.path.join(tmp, name.split()[0] + ".s1p")
    open(p, "w").write(f"# GHz S RI R 50\n0.1 {s} 0\n40 {s} 0\n")
    blocks[name] = res.connect(p, [("P2", 1)])

tri = Triangulation(mesh.nodes[:, 0] / mm, mesh.nodes[:, 1] / mm, mesh.tris)
wA = np.repeat(mesh.area, 3)


def nodal(v):  # area-weighted nodal average for smooth (Gouraud) shading
    num = np.bincount(mesh.tris.ravel(), weights=np.repeat(v, 3) * wA, minlength=len(mesh.nodes))
    den = np.bincount(mesh.tris.ravel(), weights=wA, minlength=len(mesh.nodes))
    return num / den


mags = {k: np.linalg.norm(sol.current_density(net.currents(0, "P1")[:, None])[:, :, 0], axis=1)
        for k, net in blocks.items()}
vmax = max(m.max() for m in mags.values())
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
for ax, (k, mag) in zip(axes, mags.items()):
    pc = ax.tripcolor(tri, nodal(mag), shading="gouraud", cmap="inferno", norm=LogNorm(vmax * 3e-3, vmax))
    s11 = blocks[k].S[0, 0, 0]
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.set(xlim=(-13.5, 13.5), ylim=(-1.5, 13.2), xlabel="x [mm]", ylabel="y [mm]")
    ax.set_title(f"port 2: {k}  (|S11| = {abs(s11):.2f})")
fig.colorbar(pc, ax=axes, label="|J|  [A/m per 1 A incident]", shrink=0.85)
fig.suptitle(f"Notch filter at {f[0] / GHz:.1f} GHz, wave into port 1; port 2 terminated by an "
             "S-parameter block (one EM solve, three terminations)", fontsize=11)
out = os.path.join(RESULTS, "sparams_currents.png")
fig.savefig(out, dpi=120)
print("saved", out)
