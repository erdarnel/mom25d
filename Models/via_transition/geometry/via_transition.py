"""Top microstrip -> via -> buried line: two 0.254 mm RO4350B-like layers
(eps_r 3.66, tand 0.0037, copper), as Engine/tests/test_multilayer.py and
its openEMS reference."""

from mom25d import Dielectric, Layout, Stackup, mm

ER, H1, H2 = 3.66, 0.254 * mm, 0.254 * mm
WT, WI = 1.1122 * mm, 0.45 * mm
REF, FEED = 3 * mm, 12 * mm
DRILL, PAD = 0.3 * mm, 0.6 * mm


def layout():
    lay = Layout()
    lay.add_rect(-REF, -WT / 2, 0, WT / 2, layer="top")
    lay.add_rect(0, -WI / 2, REF, WI / 2, layer="inner")
    lay.add_port("P1", (-REF, 0), "-x", WT, FEED, layer="top")
    lay.add_port("P2", (REF, 0), "+x", WI, FEED, layer="inner")
    lay.add_via(0, 0, drill=DRILL, pad=PAD, top="top", bottom="inner")
    return lay


def stackup():
    return Stackup([Dielectric(H1, ER, 0.0037), Dielectric(H2, ER, 0.0037)], {"inner": 1, "top": 2},
                   sigma=5.8e7)
