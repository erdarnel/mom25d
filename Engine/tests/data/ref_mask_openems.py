"""openEMS reference for a solder-mask covered microstrip (tests/test_odb.py).

50-ohm microstrip (W 0.56 mm) on 0.254 mm eps_r 3.66, 20 mm long between
the reference planes; case 'mask': a 25 um planar layer of eps_r 3.5 over
everything (as mom25d models solder mask), case 'bare': without it.
Writes mask_<case>_openems.txt: f, re/im S11, re/im S21 (50 ohm).
Run in ~/opt/openEMS/venv (SIM_DIR=<local tmp> for the field dumps).
"""
import os
import sys

import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

case = sys.argv[1] if len(sys.argv) > 1 else "mask"
here = os.path.dirname(os.path.abspath(__file__))
unit = 1e-6
ER, H, W = 3.66, 254.0, 560.0
TM, ERM = 25.0, 3.5
REF, L = 10000.0, 25000.0
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
ext = 5000.0
mesh.AddLine("x", [-L, L])
mesh.AddLine("y", [0, *(W / 2 + third), *(-W / 2 - third), -ext, ext])
mesh.SmoothMeshLines("x", res / 3)
mesh.SmoothMeshLines("y", res / 3)
mesh.AddLine("z", np.r_[np.linspace(0, H, 11), H + TM / 2, H + TM, 3000])
mesh.SmoothMeshLines("z", res / 2)

sub = CSX.AddMaterial("sub", epsilon=ER)
sub.AddBox([-L, -ext, 0], [L, ext, H])
if case == "mask":
    msk = CSX.AddMaterial("mask", epsilon=ERM)
    msk.AddBox([-L, -ext, H], [L, ext, H + TM])
pec = CSX.AddMetal("PEC")
port = [FDTD.AddMSLPort(1, pec, [-L, -W / 2, H], [0, W / 2, 0], "x", "z", excite=-1,
                        FeedShift=2000, MeasPlaneShift=L - REF, priority=10),
        FDTD.AddMSLPort(2, pec, [L, -W / 2, H], [0, W / 2, 0], "x", "z",
                        MeasPlaneShift=L - REF, priority=10)]
print("cells", mesh.GetQtyLines("x"), mesh.GetQtyLines("y"), mesh.GetQtyLines("z"))
sim = os.path.join(os.environ.get("SIM_DIR", here), "sim_mask_" + case)
FDTD.Run(sim, cleanup=True, verbose=0)
f = np.linspace(0.5e9, f_max, 59)
for p in port:
    p.CalcPort(sim, f, ref_impedance=50)
s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc
np.savetxt(os.path.join(here, f"mask_{case}_openems.txt"), np.c_[f, s11.real, s11.imag, s21.real, s21.imag],
           header="f re11 im11 re21 im21 (50 ohm)")
print("done")
