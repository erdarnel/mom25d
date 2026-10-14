"""Inner-plane transition (as Engine/tests/test_planes.py): top microstrip
(W 0.56 mm) over plane L2 -> via through a 0.7 mm antipad -> stripline
(W 0.24 mm) between L2 and ground, four stitching vias L2 -> gnd. Three
0.254 mm RO4350B-like layers (eps_r 3.66)."""

import numpy as np

from mom25d import Dielectric, Layout, Stackup, mm

H = 0.254 * mm
WT, WI, DRILL, PAD, ANTI = 0.56 * mm, 0.24 * mm, 0.3 * mm, 0.6 * mm, 0.7 * mm


def layout():
    lay = Layout()
    lay.add_plane("L2")
    lay.add_rect(-3 * mm, -WT / 2, 0, WT / 2, layer="top")
    lay.add_rect(0, -WI / 2, 3 * mm, WI / 2, layer="inner")
    lay.add_port("P1", (-3 * mm, 0), "-x", WT, 12 * mm, layer="top")
    lay.add_port("P2", (3 * mm, 0), "+x", WI, 12 * mm, layer="inner")
    lay.add_via(0, 0, drill=DRILL, pad=PAD, top="top", bottom="inner", antipad=ANTI)
    for a in np.pi / 4 + np.arange(4) * np.pi / 2:
        lay.add_via(1 * mm * np.cos(a), 1 * mm * np.sin(a), drill=DRILL, pad=PAD, top="L2", bottom="gnd")
    return lay


def stackup():
    return Stackup([Dielectric(H, 3.66)] * 3, {"inner": 1, "top": 3}, planes={"L2": 2})
