"""openEMS reference for mom25d multilayer validation.

case 'transition': two 0.254 mm RO4350B-like layers (eps_r 3.66) on a ground
plane, metal on the inner interface (z = 0.254 mm) and on top (z = 0.508 mm).
Top microstrip (W 1.112 mm) from port 1 to a via pad at x = 0; via (drill
0.3 mm, pads 0.6 mm) down to the inner layer; buried line (W 0.45 mm) to port
2. S (50 ohm) referenced to x = -3 / +3 mm. Writes transition.txt:
f, re/im S11, re/im S21, re/im S22.
"""
import os, sys
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

case = sys.argv[1] if len(sys.argv) > 1 else "transition"
here = os.path.dirname(os.path.abspath(__file__))
unit = 1e-6
ER, H1, H2 = 3.66, 254.0, 254.0
ZI, ZT = H1, H1 + H2
WT, WI = 1112.2, 450.0
DRILL, PAD = 300.0, 600.0
REF, L = 3000.0, 20000.0
f_max = 15e9

FDTD = openEMS()
FDTD.SetGaussExcite(f_max / 2, f_max / 2)
FDTD.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])
FDTD.SetEndCriteria(1e-5)
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
res = C0 / (f_max * np.sqrt(ER)) / unit / 50
fine = 40.0
third = np.array([2 * fine / 3, -fine / 3])
ext = 8000.0
mesh.AddLine("x", [-L, L, 0, -DRILL / 2, DRILL / 2, -PAD / 2, PAD / 2])
mesh.AddLine("y", [0, *(WT / 2 + third), *(-WT / 2 - third), *(WI / 2 + third), *(-WI / 2 - third),
                   -DRILL / 2, DRILL / 2, -ext, ext])
mesh.SmoothMeshLines("x", res / 3)
mesh.SmoothMeshLines("y", res / 3)
mesh.AddLine("z", np.linspace(0, ZT, 11))
mesh.AddLine("z", 3000)
mesh.SmoothMeshLines("z", res / 2)

sub = CSX.AddMaterial("sub", epsilon=ER)
sub.AddBox([-L, -ext, 0], [L, ext, ZT])
pec = CSX.AddMetal("PEC")
port = [FDTD.AddMSLPort(1, pec, [-L, -WT / 2, ZT], [0, WT / 2, 0], "x", "z", excite=-1,
                        FeedShift=2000, MeasPlaneShift=L - REF, priority=10),
        FDTD.AddMSLPort(2, pec, [L, -WI / 2, ZI], [0, WI / 2, 0], "x", "z",
                        MeasPlaneShift=L - REF, priority=10)]
pec.AddCylinder([0, 0, ZI], [0, 0, ZT], DRILL / 2, priority=20)
for z in (ZI, ZT):
    pec.AddCylinder([0, 0, z - 1e-3], [0, 0, z + 1e-3], PAD / 2, priority=20)
print("cells", mesh.GetQtyLines("x"), mesh.GetQtyLines("y"), mesh.GetQtyLines("z"))
sim = os.path.join(here, "sim_" + case)
FDTD.Run(sim, cleanup=True, verbose=0)
f = np.linspace(0.5e9, f_max, 59)
for p in port:
    p.CalcPort(sim, f, ref_impedance=50)
s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc
np.savetxt(os.path.join(here, case + ".txt"), np.c_[f, s11.real, s11.imag, s21.real, s21.imag],
           header="f re11 im11 re21 im21 (50 ohm)")
print("done")
