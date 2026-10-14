"""openEMS reference for lumped components (tests/test_lumped.py).

50-ohm microstrip (W 0.56 mm, 0.254 mm eps_r 3.66), 5 mm either side of a
0.5 mm gap at x = 0. case 'seriesC': a 1 pF lumped capacitor fills the gap
(openEMS LumpedElement, direction x, W wide, at the strip). S (50 ohm)
referenced to x = -5 / +5 mm. Writes lumped_<case>_openems.txt:
f, re/im S11, re/im S21. Run in ~/opt/openEMS/venv (SIM_DIR=<local tmp>).
"""
import os
import sys

import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

case = sys.argv[1] if len(sys.argv) > 1 else "seriesC"
here = os.path.dirname(os.path.abspath(__file__))
unit = 1e-6
ER, H, W, G = 3.66, 254.0, 560.0, 500.0
REF, L = 5000.0, 20000.0
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
ext = 6000.0
mesh.AddLine("x", [-L, L, -G / 2, G / 2, *np.linspace(-G / 2, G / 2, 6)])
mesh.AddLine("y", [0, *(W / 2 + third), *(-W / 2 - third), -ext, ext])
mesh.SmoothMeshLines("x", res / 3)
mesh.SmoothMeshLines("y", res / 3)
mesh.AddLine("z", np.r_[np.linspace(0, H, 11), 3000])
mesh.SmoothMeshLines("z", res / 2)

sub = CSX.AddMaterial("sub", epsilon=ER)
sub.AddBox([-L, -ext, 0], [L, ext, H])
pec = CSX.AddMetal("PEC")
port = [FDTD.AddMSLPort(1, pec, [-L, -W / 2, H], [-G / 2, W / 2, 0], "x", "z", excite=-1,
                        FeedShift=2000, MeasPlaneShift=L - REF, priority=10),
        FDTD.AddMSLPort(2, pec, [L, -W / 2, H], [G / 2, W / 2, 0], "x", "z",
                        MeasPlaneShift=L - REF, priority=10)]
if case == "seriesC":
    lc = CSX.AddLumpedElement("C1", ny=0, caps=True, C=1e-12)
    lc.AddBox([-G / 2, -W / 2, H], [G / 2, W / 2, H])
print("cells", mesh.GetQtyLines("x"), mesh.GetQtyLines("y"), mesh.GetQtyLines("z"))
sim = os.path.join(os.environ.get("SIM_DIR", here), "sim_lumped_" + case)
FDTD.Run(sim, cleanup=True, verbose=0)
f = np.linspace(0.5e9, f_max, 59)
for p in port:
    p.CalcPort(sim, f, ref_impedance=50)
s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc
np.savetxt(os.path.join(here, f"lumped_{case}_openems.txt"), np.c_[f, s11.real, s11.imag, s21.real, s21.imag],
           header="f re11 im11 re21 im21 (50 ohm)")
print("done")
