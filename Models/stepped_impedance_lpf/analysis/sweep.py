"""5th-order Butterworth stepped-impedance low-pass filter (fc = 3 GHz)
on 0.508 mm RO4003C, designed with the textbook formulas, then analysed
with the 2.5D MoM. Also plots the surface current at two frequencies."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")

from mom25d import GHz, Solver, mm  # noqa: E402
from lpf import FC as fc, layout, sections, substrate  # noqa: E402

print("widths/lengths [mm]:", [(round(w / mm, 3), round(l / mm, 3)) for w, l in sections()])

mesh = layout().mesh(max_cell=0.6 * mm, edge_cell=0.06 * mm)

sol = Solver(mesh, substrate())
f = np.linspace(0.25, 9, 25) * GHz
res = sol.sweep(f, keep_currents=True)
S = res.renormalize(50)
res.write_touchstone(os.path.join(RESULTS, "lpf.s2p"), zref=50)

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1])
ax = fig.add_subplot(gs[0])
ax.plot(f / GHz, res.db(1, 0, S), "-o", ms=3, label="S21")
ax.plot(f / GHz, res.db(0, 0, S), "-s", ms=3, label="S11")
ax.axvline(fc / GHz, color="k", lw=0.8, ls=":")
ax.set(xlabel="frequency [GHz]", ylabel="dB", ylim=(-50, 2),
       title="Stepped-impedance LPF, N=5 Butterworth, fc = 3 GHz (with dielectric + conductor loss)")
ax.grid(alpha=0.3)
ax.legend()

for i, fsel in enumerate((1.5 * GHz, 6.5 * GHz)):
    k = np.argmin(abs(f - fsel))
    x_m = res.matched_currents(k, 0)  # unit wave into port 1, port 2 matched
    J = sol.current_density(x_m[:, None])[:, :, 0]
    mag = np.linalg.norm(np.abs(J), axis=1)
    a = fig.add_subplot(gs[1 + i])
    pc = mesh.plot(a, values=mag, cmap="inferno", norm=LogNorm(mag.max() * 1e-3, mag.max()))
    pc.set_edgecolor("none")
    a.set_facecolor("#dddddd")
    a.set_title(f"|J| at {f[k] / GHz:.2f} GHz, wave incident on port 1, port 2 matched "
                "(feed lines beyond the gaps are not physical)", fontsize=9)
    fig.colorbar(pc, ax=a, label="|J| (rel.)", pad=0.01, aspect=10)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "lpf.png"), dpi=130)
print("saved lpf.png / lpf.s2p")
