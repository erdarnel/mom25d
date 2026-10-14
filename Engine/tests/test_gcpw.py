import os

import numpy as np
import pytest

from mom25d import CoplanarGround, Layout, Solver, Substrate, mm

shapely = pytest.importorskip("shapely")
DATA = os.path.join(os.path.dirname(__file__), "data")
ER, H = 3.66, 0.254 * mm
W, G, WP = 0.5561 * mm, 0.2 * mm, 1.2 * mm  # tests/data/ref_vias_openems.py gcpw
REF, FEED = 5 * mm, 6 * mm


def gcpw_layout(ports_gcpw=True):
    """10 mm GCPW line: pour (with via fence) only between the reference
    planes; outside, the GCPW feeds carry the grounds."""
    lay = Layout()
    lay.add_rect(-REF, -W / 2, REF, W / 2)
    gnd = CoplanarGround(G, WP, via_pitch=1 * mm, via_drill=0.3 * mm, via_pad=0.5 * mm) if ports_gcpw else None
    lay.add_port("P1", (-REF, 0), "-x", W, FEED, ground=gnd)
    lay.add_port("P2", (REF, 0), "+x", W, FEED, ground=gnd)
    y1 = W / 2 + G + WP
    lay.add_pour((-REF, -y1, REF, y1), G, via_pitch=1 * mm, via_drill=0.3 * mm, via_pad=0.5 * mm)
    return lay


def test_gcpw_feed_geometry():
    lay = gcpw_layout()
    p = lay.ports[0]
    strips = [shapely.geometry.Polygon(c) for c in p.ground_strips()]
    sig = shapely.geometry.Polygon(p.corners())
    assert len(strips) == 2
    for st in strips:
        assert st.distance(sig) == pytest.approx(G, rel=1e-9)
        assert st.area == pytest.approx(WP * FEED, rel=1e-9)
    vias = p.ground_vias()
    assert len(vias) == 2 * int(FEED / (1 * mm))
    assert all(any(st.contains(shapely.geometry.Point(v.xy).buffer(v.pad / 2 * 0.99)) for st in strips)
               for v in vias)
    # the pour joins the feed grounds at the reference plane (no clearance cut)
    (pour, _), = lay.resolved_pours()
    for st in strips:
        assert pour.distance(st) < 1e-12
    # asymmetric: ground on the left (+t) side only
    q = Layout().add_port("Q", (0, 0), "+x", W, FEED, ground=CoplanarGround((G, None), (WP, 0)))
    (c,) = q.ground_strips()
    assert min(y for _, y in c) == pytest.approx(W / 2 + G)


def test_gcpw_and_microstrip_ports_do_not_share_a_line():
    lay = Layout()
    lay.add_rect(-REF, -W / 2, REF, W / 2)
    lay.add_port("P1", (-REF, 0), "-x", W, FEED, ground=CoplanarGround(G, WP, via_pitch=1 * mm))
    lay.add_port("P2", (REF, 0), "+x", W, FEED)
    sol = Solver(lay.mesh(0.3 * mm, 0.05 * mm, kind="tri"), Substrate(ER, H), verbose=False)
    assert len(sol.groups) == 2


def test_port_grounds_from_pour():
    from shapely.geometry import box

    from mom25d.odb import port_grounds

    # trace along x crossing the right window edge, pour above (gap 0.3) and
    # below (gap 0.5), 0.8 mm wide; a second trace without pour on the left
    pour = box(5, 0.2 + 0.3, 10, 0.2 + 0.3 + 0.8).union(box(5, -0.2 - 0.5 - 0.8, 10, -0.2 - 0.5))
    ports = [((10.0, 0.0), "+x", 0.4), ((0.0, 3.0), "-x", 0.4)]
    g = port_grounds(pour, (0, 10, -5, 5), ports)
    (gaps, widths), none = g
    assert none is None
    # +x feed: left (+t) = +y
    assert gaps == pytest.approx((0.3, 0.5)) and widths == pytest.approx((0.8, 0.8))


def test_gcpw_ports_vs_openems():
    """GCPW feeds reproduce the pour-everywhere line (eeff 2.72-2.76, Z0
    47.8-48.5 ohm, Models/gcpw_pour/analysis/vs_openems.py) and openEMS; plain
    microstrip feeds on the same windowed pour give eeff 2.86, Z0 50.6."""
    d = np.loadtxt(os.path.join(DATA, "gcpw_pour_openems.txt"))[[10, 42]]  # 3, 11 GHz
    f, s11r, s21r = d[:, 0], d[:, 1] + 1j * d[:, 2], d[:, 3] + 1j * d[:, 4]
    r = Solver(gcpw_layout().mesh(0.25 * mm, 0.05 * mm, kind="tri"), Substrate(ER, H),
               verbose=False).sweep(f)
    np.testing.assert_allclose(r.eeff[:, 0], [2.725, 2.749], rtol=3e-3)
    np.testing.assert_allclose(r.z0_line[:, 0].real, [47.8, 48.2], rtol=5e-3)
    S = r.renormalize(50)
    assert np.all(np.abs(np.angle(S[:, 1, 0] / s21r, deg=True)) < 2.5)
    np.testing.assert_allclose(np.abs(S[:, 0, 0]), np.abs(s11r), atol=0.015)
