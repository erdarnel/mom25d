"""Grounded coplanar waveguide made from a via-stitched copper pour, vs openEMS.

50-ohm microstrip (RO4350B, h = 0.254 mm) with a GND pour on both sides
(gap 0.2 mm, 1.2 mm wide, via fences 1 mm pitch, 0.3 mm drill / 0.5 mm pad),
S referenced (50 ohm) 5 mm either side of the centre. Reference:
Engine/tests/data/ref_vias_openems.py gcpw. ~5 min (N ~ 5000 incl. 88 vias).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")
REFDATA = os.path.join(here, "..", "..", "..", "Engine", "tests", "data")

from gcpw_pour import layout, substrate  # noqa: E402
from mom25d import GHz, Solver, mm  # noqa: E402

mesh = layout().mesh(max_cell=0.25 * mm, edge_cell=0.05 * mm)

f = np.linspace(1, 15, 15) * GHz
sol = Solver(mesh, substrate())
res = sol.sweep(f)
S = res.renormalize(50)
ref = np.loadtxt(os.path.join(REFDATA, "gcpw_pour_openems.txt"))
fr, s11, s21 = ref[:, 0], ref[:, 1] + 1j * ref[:, 2], ref[:, 3] + 1j * ref[:, 4]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
mesh.plot(ax[0])
for v in mesh.vias:
    ax[0].add_patch(plt.Circle(np.array(v.xy) / mm, v.drill / 2 / mm, color="r"))
ax[0].set_title(f"mesh: {mesh.n_basis} RWG + {mesh.n_vias} vias")
ax[1].plot(fr / GHz, 20 * np.log10(abs(s11)), "k-", label="|S11| openEMS")
ax[1].plot(f / GHz, res.db(0, 0, S), "ro", label="|S11| mom25d")
ax[1].set(xlabel="GHz", ylabel="dB (50 Ω)", ylim=(-50, 0))
ax[1].legend()
ax[2].plot(fr / GHz, np.unwrap(np.angle(s21)) * 180 / np.pi, "k-", label="∠S21 openEMS")
ax[2].plot(f / GHz, np.unwrap(np.angle(S[:, 1, 0])) * 180 / np.pi, "ro", label="∠S21 mom25d")
ax[2].set(xlabel="GHz", ylabel="deg")
ax[2].legend()
for a in ax[1:]:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "gcpw_pour.png"), dpi=120)
print("eeff", np.round(res.eeff[:, 0], 3), "\nZ0 line", np.round(res.z0_line[:, 0].real, 2))
print("saved gcpw_pour.png")
