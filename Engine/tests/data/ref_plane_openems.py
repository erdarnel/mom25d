"""openEMS reference for mom25d inner PEC planes (tests/test_planes.py).

Three 0.254 mm layers (eps_r 3.66) on a ground plane; a solid PEC plane L2 at
z = 0.508 mm with a 0.7 mm antipad around the via; inner line at z = 0.254 mm
(stripline between gnd and L2), top microstrip at z = 0.762 mm (over L2).
Top line (W 0.56 mm) from port 1 to the via at x = 0; via (drill 0.3 mm, pads
0.6 mm) from top through the antipad down to the inner line (W 0.24 mm) to
port 2. S (50 ohm) referenced to x = -3 / +3 mm.

case 'plane'        : no stitching vias (the via leaks into the L2/gnd
                      parallel plate; PML / MUR at the board edges stand in
                      for the infinite planes of the MoM model)
case 'plane_stitch' : 4 vias L2 -> gnd (drill 0.3) on a 1 mm radius at 45,
                      135, 225, 315 deg

plane_openems.txt (2026-10-03) was stopped at -45.8 dB (touch sim_plane/ABORT):
the leaked parallel-plate wave decays very slowly in the PML at grazing
incidence; -50 dB was not reached after 86k steps (41 min). plane_stitch
reached -50 dB in 36k steps (18 min).

Args: case [antipad_um [end_criterion [fine_cell_um]]], e.g.
`plane_stitch 400 1e-4` -> plane_stitch_anti400_openems.txt,
`plane_stitch 700 1e-4 20` -> plane_stitch_anti700_fine20_openems.txt
(antipad study, mesh convergence).

Run in ~/opt/openEMS/venv (SIM_DIR=<local tmp> for the field dumps); writes
<case>_openems.txt:
f, re/im S11, re/im S21.
"""
import os
import sys

import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

case = sys.argv[1] if len(sys.argv) > 1 else "plane"
# optional: antipad diameter [um] (default 700) and end criterion (default 1e-5)
anti_arg = float(sys.argv[2]) if len(sys.argv) > 2 else None
end_crit = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
# optional: refined mesh -- edge-refinement spacing `fine` [um] (default 40),
# plus the same third-rule lines at the drill / pad / antipad radii and
# 10 z cells per layer (convergence check of the antipad region)
fine_cell = float(sys.argv[4]) if len(sys.argv) > 4 else None
here = os.path.dirname(os.path.abspath(__file__))
unit = 1e-6
ER, H = 3.66, 254.0
ZI, ZP, ZT = H, 2 * H, 3 * H
WT, WI = 560.0, 240.0
DRILL, PAD, ANTI = 300.0, 600.0, 700.0
if anti_arg:
    ANTI = anti_arg
REF, L = 3000.0, 20000.0
RS = 1000.0  # stitching via radius
f_max = 15e9
stitch = []
if case.startswith("plane_stitch"):
    stitch = [(RS * np.cos(a), RS * np.sin(a)) for a in np.pi / 4 + np.arange(4) * np.pi / 2]

FDTD = openEMS()
FDTD.SetGaussExcite(f_max / 2, f_max / 2)
FDTD.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "PEC", "MUR"])
FDTD.SetEndCriteria(end_crit)
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
res = C0 / (f_max * np.sqrt(ER)) / unit / 50
fine = fine_cell or 40.0
third = np.array([2 * fine / 3, -fine / 3])
ext = 8000.0
xs = [-L, L, 0, -DRILL / 2, DRILL / 2, -PAD / 2, PAD / 2, -ANTI / 2, ANTI / 2]
ys = [0, *(WT / 2 + third), *(-WT / 2 - third), *(WI / 2 + third), *(-WI / 2 - third),
      -DRILL / 2, DRILL / 2, -ANTI / 2, ANTI / 2, -ext, ext]
for x, y in stitch:
    xs += [x - DRILL / 2, x, x + DRILL / 2]
    ys += [y - DRILL / 2, y, y + DRILL / 2]
if fine_cell:
    # third-rule lines around the circular edges too (drill, pad, antipad);
    # (an earlier variant overlaid a uniform 20 um box: near-coincident lines,
    # 3 um cells, and a transition that did not conduct -- |S11| = 1)
    for r in (DRILL / 2, PAD / 2, ANTI / 2):
        for lst in (xs, ys):
            lst += [r + third[0], r + third[1], -r - third[0], -r - third[1]]
mesh.AddLine("x", xs)
mesh.AddLine("y", ys)
mesh.SmoothMeshLines("x", res / 3)
mesh.SmoothMeshLines("y", res / 3)
# refined: uniform z cells that put every metal level (H, 2H, 3H) on a line
mesh.AddLine("z", np.linspace(0, ZT, 16) if not fine_cell else np.linspace(0, ZT, 31))
mesh.AddLine("z", 3000)
mesh.SmoothMeshLines("z", res / 2)

sub = CSX.AddMaterial("sub", epsilon=ER)
sub.AddBox([-L, -ext, 0], [L, ext, ZT], priority=1)
pec = CSX.AddMetal("PEC")
pec.AddBox([-L, -ext, ZP], [L, ext, ZP], priority=5)  # L2
sub.AddCylinder([0, 0, ZP - 1], [0, 0, ZP + 1], ANTI / 2, priority=6)  # antipad
port = [FDTD.AddMSLPort(1, pec, [-L, -WT / 2, ZT], [0, WT / 2, ZP], "x", "z", excite=-1,
                        FeedShift=2000, MeasPlaneShift=L - REF, priority=10),
        FDTD.AddMSLPort(2, pec, [L, -WI / 2, ZI], [0, WI / 2, 0], "x", "z",
                        MeasPlaneShift=L - REF, priority=10)]
pec.AddCylinder([0, 0, ZI], [0, 0, ZT], DRILL / 2, priority=20)
for z in (ZI, ZT):
    pec.AddCylinder([0, 0, z - 1e-3], [0, 0, z + 1e-3], PAD / 2, priority=20)
for x, y in stitch:
    pec.AddCylinder([x, y, 0], [x, y, ZP], DRILL / 2, priority=20)
print("cells", mesh.GetQtyLines("x"), mesh.GetQtyLines("y"), mesh.GetQtyLines("z"))
tag = case + (f"_anti{ANTI:.0f}" if anti_arg else "") + (f"_fine{fine_cell:.0f}" if fine_cell else "")
sim = os.path.join(os.environ.get("SIM_DIR", here), "sim_" + tag)  # keep off synced drives
FDTD.Run(sim, cleanup=True, verbose=0)
f = np.linspace(0.5e9, f_max, 59)
for p in port:
    p.CalcPort(sim, f, ref_impedance=50)
s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc
np.savetxt(os.path.join(here, tag + "_openems.txt"), np.c_[f, s11.real, s11.imag, s21.real, s21.imag],
           header="f re11 im11 re21 im21 (50 ohm)")
print("done")
