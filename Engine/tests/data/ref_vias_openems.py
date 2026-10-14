"""openEMS references for mom25d via / pour validation.

case 'via'  : 50-ohm microstrip with one grounding via (drill 0.3, pad = W) at x=0,
              S referenced (50 ohm) to x = -3 / +3 mm.
case 'gcpw' : same line with ground pours on both sides (gap G, width WP),
              via fences from mom25d.Pour.resolve, refs at x = -5 / +5 mm.
Writes <case>.txt with columns f, re/im S11, re/im S21.
"""
import os, sys
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
case = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
unit = 1e-6
W, H, ER = 556.1, 254.0, 3.66
DRILL = 300.0
f_max = 15e9
REF = 3000.0 if case == "via" else 5000.0
L = 20000.0  # half length of the line (port ends)
G, WP, PITCH, PAD = 200.0, 1200.0, 1000.0, 500.0

vias = []  # (x, y, r_drill, r_pad)
pours = []
if case == "via":
    vias.append((0.0, 0.0, DRILL / 2, W / 2))
else:
    y0 = W / 2 + G
    pours = [(-L, y0, L, y0 + WP), (-L, -y0 - WP, L, -y0)]
    # fence positions from mom25d.Pour.resolve (written with the mom25d venv)
    vias = [(x, y, DRILL / 2, PAD / 2) for x, y in np.loadtxt(os.path.join(here, "gcpw_vias.txt"))]

FDTD = openEMS()
FDTD.SetGaussExcite(f_max / 2, f_max / 2)
FDTD.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
res = C0 / (f_max * np.sqrt(ER)) / unit / 50
fine = 40.0
third = np.array([2 * fine / 3, -fine / 3])

ylines = [0, *(W / 2 + third), *(-W / 2 - third)]
xlines = [-L, L, 0]
for x, y, rd, rp in vias:
    xlines += [x - rd, x, x + rd]
    ylines += [y - rd, y, y + rd]
for b in pours:
    ylines += [b[1], b[3]]
ext = (W / 2 + G + WP if pours else 0) + 15 * W
ylines += [-ext, ext]
mesh.AddLine("x", xlines)
mesh.AddLine("y", ylines)
mesh.SmoothMeshLines("x", res / 3)
mesh.SmoothMeshLines("y", res / 3)
mesh.AddLine("z", np.linspace(0, H, 6))
mesh.AddLine("z", 3000)
mesh.SmoothMeshLines("z", res / 2)

sub = CSX.AddMaterial("sub", epsilon=ER)
sub.AddBox([-L, -ext, 0], [L, ext, H])
pec = CSX.AddMetal("PEC")
port = [FDTD.AddMSLPort(1, pec, [-L, -W / 2, H], [0, W / 2, 0], "x", "z", excite=-1,
                        FeedShift=2000, MeasPlaneShift=L - REF, priority=10),
        FDTD.AddMSLPort(2, pec, [L, -W / 2, H], [0, W / 2, 0], "x", "z",
                        MeasPlaneShift=L - REF, priority=10)]
for b in pours:
    pec.AddBox([b[0], b[1], H], [b[2], b[3], H], priority=10)
for x, y, rd, rp in vias:
    pec.AddCylinder([x, y, 0], [x, y, H], rd, priority=20)
    pec.AddCylinder([x, y, H], [x, y, H], rp, priority=20)
print("cells", mesh.GetQtyLines("x"), mesh.GetQtyLines("y"), mesh.GetQtyLines("z"), "vias", len(vias))
sim = os.path.join(here, "sim_" + case)
FDTD.SetEndCriteria(1e-5)
FDTD.Run(sim, cleanup=True, verbose=0)
f = np.linspace(0.5e9, f_max, 59)
for p in port:
    p.CalcPort(sim, f, ref_impedance=50)
s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc
np.savetxt(os.path.join(here, case + ".txt"), np.c_[f, s11.real, s11.imag, s21.real, s21.imag],
           header="f re11 im11 re21 im21 (50 ohm)")
print("done")
