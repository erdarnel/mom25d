"""5th-order Butterworth stepped-impedance low-pass filter (fc = 3 GHz) on
0.508 mm RO4003C, sections from the textbook formulas (100 / 20 ohm lines,
50 ohm feeds)."""

from mom25d import GHz, Layout, Substrate, analytic, mm

ER, H = 3.55, 0.508 * mm
FC, R0, ZHI, ZLO = 3 * GHz, 50.0, 100.0, 20.0
G = [0.618, 1.618, 2.000, 1.618, 0.618]  # Butterworth N=5


def sections():
    """[(width, length)] of the five line sections."""
    whi, wlo = analytic.width_for_z0(ZHI, H, ER), analytic.width_for_z0(ZLO, H, ER)
    out = []
    for k, gk in enumerate(G):
        # series L -> high-impedance line, shunt C -> low-impedance line
        W, bl = (whi, gk * R0 / ZHI) if k % 2 == 0 else (wlo, gk * ZLO / R0)
        out.append((W, bl / analytic.beta(W, H, ER, FC)))
    return out


def length():
    return sum(L for _, L in sections())


def layout():
    lay = Layout()
    x = 0.0
    for W, L in sections():
        lay.add_rect(x, -W / 2, x + L, W / 2)
        x += L
    W0 = analytic.width_for_z0(R0, H, ER)
    lay.add_port("P1", (0, 0), "-x", W0, 16 * mm)
    lay.add_port("P2", (x, 0), "+x", W0, 16 * mm)
    return lay


def substrate():
    return Substrate(ER, H, tand=0.0027, sigma=5.8e7, t=35e-6)
