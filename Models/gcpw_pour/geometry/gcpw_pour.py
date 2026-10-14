"""Grounded coplanar waveguide made from a via-stitched copper pour:
50-ohm microstrip (RO4350B, h = 0.254 mm) with a GND pour on both sides
(gap 0.2 mm, 1.2 mm wide, via fences 1 mm pitch, 0.3 mm drill / 0.5 mm
pad), reference planes 5 mm either side of the centre."""

from mom25d import Layout, Substrate, mm

ER, H = 3.66, 0.254 * mm
W, G, WP = 0.5561 * mm, 0.2 * mm, 1.2 * mm
REF, FEED = 5 * mm, 6 * mm


def layout():
    lay = Layout()
    lay.add_rect(-REF, -W / 2, REF, W / 2)
    lay.add_port("P1", (-REF, 0), "-x", W, FEED)
    lay.add_port("P2", (REF, 0), "+x", W, FEED)
    E, y1 = REF + FEED, W / 2 + G + WP
    lay.add_pour((-E, -y1, E, y1), G, via_pitch=1 * mm, via_drill=0.3 * mm, via_pad=0.5 * mm)
    return lay


def substrate():
    return Substrate(ER, H)
