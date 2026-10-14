"""Surface current density on the open-stub notch filter, below and at the
notch, for a unit wave incident on port 1 with port 2 matched."""

import os
import sys

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

sol = Solver(mesh, substrate())
freqs = np.array([2.0, 3.67]) * GHz
res = sol.sweep(freqs, keep_currents=True)

# triangle-centroid J -> area-weighted nodal average, for smooth (Gouraud) shading
tri = Triangulation(mesh.nodes[:, 0] / mm, mesh.nodes[:, 1] / mm, mesh.tris)
wA = np.repeat(mesh.area, 3)


def nodal(v):
    num = np.bincount(mesh.tris.ravel(), weights=np.repeat(v, 3) * wA, minlength=len(mesh.nodes))
    den = np.bincount(mesh.tris.ravel(), weights=wA, minlength=len(mesh.nodes))
    return num / den


fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True,
                         gridspec_kw=dict(height_ratios=[1, 1.1]))
views = [((-13.5, 13.5), (-1.5, 13.2), 13, 0.0025), ((-2.2, 2.2), (-0.9, 2.3), 1, 0.004)]
for fi in range(len(freqs)):
    J = sol.current_density(res.matched_currents(fi, 0)[:, None])[:, :, 0]  # (Nt,2) complex
    mag = np.sqrt(np.sum(np.abs(J) ** 2, axis=1))
    vmax = mag.max()
    s21 = 20 * np.log10(abs(res.renormalize(50)[fi, 1, 0]))
    for row, ((x0, x1), (y0, y1), stride, width) in enumerate(views):
        ax = axes[row, fi]
        pc = ax.tripcolor(tri, nodal(mag), shading="gouraud", cmap="inferno",
                          norm=LogNorm(vmax * 3e-3, vmax))
        # direction of the instantaneous current (incident-wave phase 0 at port 1)
        c = mesh.centroid / mm
        inside = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)
        sel = inside & (mag > vmax * 0.01)
        sel &= np.cumsum(sel) % stride == 0
        Jr = J[sel].real
        n = np.linalg.norm(Jr, axis=1) + 1e-30
        ax.quiver(c[sel, 0], c[sel, 1], Jr[:, 0] / n, Jr[:, 1] / n, color="cyan",
                  scale=45 if row == 0 else 30, width=width, alpha=0.85)
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")
        ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="x [mm]", ylabel="y [mm]")
        zoom = "" if row == 0 else " — T-junction zoom"
        ax.set_title(f"|J| at {freqs[fi] / GHz:.2f} GHz (S21 = {s21:.1f} dB){zoom}")
        fig.colorbar(pc, ax=ax, label="|J|  [A/m per 1 A incident]", shrink=0.9)
fig.suptitle("Open-stub notch filter: wave incident on port 1 (left), port 2 matched. "
             "Arrows: Re J direction. Feed ends beyond the gaps are not physical.", fontsize=10)
fig.savefig(os.path.join(RESULTS, "current_density.png"), dpi=140)
print("saved current_density.png, N =", mesh.n_basis)
