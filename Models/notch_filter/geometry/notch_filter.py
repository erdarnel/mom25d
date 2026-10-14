"""Open-stub notch filter (openEMS MSL_NotchFilter tutorial geometry):
0.6 mm line on 0.254 mm eps_r 3.66, 12 mm open stub. Mesh settings are
chosen per analysis."""

from mom25d import Layout, Substrate, mm

ER, H, W, LS = 3.66, 0.254 * mm, 0.6 * mm, 12 * mm


def layout():
    lay = Layout()
    lay.add_rect(-W / 2 - 1 * mm, -W / 2, W / 2 + 1 * mm, W / 2)  # main line around the tee
    lay.add_rect(-W / 2, W / 2, W / 2, W / 2 + LS)  # open stub
    lay.add_port("P1", (-W / 2 - 1 * mm, 0), "-x", W, 12 * mm)
    lay.add_port("P2", (W / 2 + 1 * mm, 0), "+x", W, 12 * mm)
    return lay


def substrate():
    return Substrate(ER, H)
