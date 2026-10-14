import os

import numpy as np
import pytest

from mom25d import GHz, Layout, Solver, Substrate, analytic, mm

pytest.importorskip("triangle")
pytest.importorskip("shapely")

DATA = os.path.join(os.path.dirname(__file__), "data")


def notch_layout():
    W, Ls = 0.6 * mm, 12 * mm
    lay = Layout()
    lay.add_rect(-W / 2 - 1 * mm, -W / 2, W / 2 + 1 * mm, W / 2)
    lay.add_rect(-W / 2, W / 2, W / 2, W / 2 + Ls)
    lay.add_port("P1", (-W / 2 - 1 * mm, 0), "-x", W, 12 * mm)
    lay.add_port("P2", (W / 2 + 1 * mm, 0), "+x", W, 12 * mm)
    return lay


@pytest.mark.parametrize("angle,aspect", [(15.0, None), (20.0, None), (25.0, None), (20.0, 5.0)])
def test_tri_mesh_quality_and_ports(angle, aspect):
    """Isotropic (aspect = 1/tan(angle)): every angle >= min_angle. With an
    anisotropic edge band, only its cells (edge_cell x aspect*edge_cell)
    go below: smallest angle atan(1/aspect)."""
    iso = 1 / np.tan(np.radians(angle))
    lay = notch_layout()
    m = lay.mesh(0.5 * mm, 0.05 * mm, kind="tri", min_angle=angle, max_aspect=aspect or iso)
    amin, rmax = m.quality()
    if aspect is None:
        assert amin >= angle - 0.1
    else:
        assert amin >= np.degrees(np.arctan(1 / aspect)) - 0.1 and rmax < 4
    # structured feeds: every cut is a full row of mesh edges across the line
    for p in lay.ports:
        s = m.cut_positions(p)
        assert s[0] == 0 and s[-1] == pytest.approx(p.length) and np.any(np.isclose(s, p.s_gap))
        assert np.max(np.diff(s)) <= 0.05 * mm * (aspect or iso) * (1 + 1e-9)
        for si in s[1:-1]:
            idx, _ = m.cut_edges(p, si)
            assert m.edge_len[idx].sum() == pytest.approx(p.width, rel=1e-9)
    # copper area is reproduced exactly (no staircase)
    W, Ls = 0.6 * mm, 12 * mm
    area = (W + 2 * mm) * W + W * Ls + 2 * 12 * mm * W
    assert m.area.sum() == pytest.approx(area, rel=1e-9)


def test_tri_rejects_bad_angle():
    with pytest.raises(ValueError):
        notch_layout().mesh(0.5 * mm, 0.05 * mm, kind="tri", min_angle=35)


def test_tri_through_line():
    ER, H = 3.55, 0.508 * mm
    W = analytic.width_for_z0(50, H, ER)
    lay = Layout()
    lay.add_rect(0, -W / 2, 10 * mm, W / 2)
    lay.add_port("P1", (0, 0), "-x", W, 20 * mm)
    lay.add_port("P2", (10 * mm, 0), "+x", W, 20 * mm)
    f = np.array([2, 10]) * GHz
    m = lay.mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15)
    r = Solver(m, Substrate(ER, H), verbose=False).sweep(f)
    np.testing.assert_allclose(r.eeff[:, 0], analytic.eeff(W, H, ER, f), rtol=5e-3)
    z0 = r.z0_line[:, 0].real  # grid mesh: 50.55, 51.28 ohm
    assert z0[0] == pytest.approx(50, rel=0.03) and z0[0] < z0[1] < 1.025 * z0[0]
    assert np.all(np.abs(r.S[:, 0, 0]) < 5e-3)
    np.testing.assert_allclose(np.abs(r.S[:, 1, 0]), 1, atol=5e-3)
    phase = np.angle(r.S[:, 1, 0] * np.exp(1j * analytic.beta(W, H, ER, f) * 10 * mm))
    assert np.all(np.abs(phase) < np.deg2rad(0.5))


def test_tri_notch_filter_vs_openems():
    ref = np.loadtxt(os.path.join(DATA, "notch_openems.txt"))
    f = np.array([2.0, 3.6, 3.65, 3.7, 6.0]) * GHz
    m = notch_layout().mesh(0.5 * mm, 0.05 * mm, kind="tri", min_angle=15)
    r = Solver(m, Substrate(3.66, 0.254 * mm), verbose=False).sweep(f)
    s21 = r.db(1, 0, r.renormalize(50))
    ref21 = 20 * np.log10(np.interp(f, ref[:, 0], ref[:, 2]))
    band = np.abs(ref21) < 15
    assert np.max(np.abs(s21[band] - ref21[band])) < 0.3
    p = np.polyfit(f[1:4] / GHz, s21[1:4], 2)
    assert -p[1] / (2 * p[0]) == pytest.approx(3.671, rel=3e-3)


def test_tri_oblique_line():
    """A 45-degree line between two Manhattan feeds is meshed exactly (no
    staircase): lossless, matched, and its electrical length is the true
    path length."""
    from shapely.geometry import LineString

    ER, H = 3.66, 0.254 * mm
    W = 0.5561 * mm
    a = 3 * mm  # diagonal run: a in x and a in y
    lay = Layout()
    path = LineString([(-1 * mm, 0), (0, 0), (a, a), (a + 1 * mm, a)])
    lay.add_polygon(path.buffer(W / 2, cap_style="flat", join_style="mitre"))
    lay.add_port("P1", (-1 * mm, 0), "-x", W, 8 * mm)
    lay.add_port("P2", (a + 1 * mm, a), "+x", W, 8 * mm)
    m = lay.mesh(0.4 * mm, 0.06 * mm, kind="tri", min_angle=15)
    f = np.array([3, 8]) * GHz
    S = Solver(m, Substrate(ER, H), verbose=False).sweep(f).S
    assert np.all(np.abs(np.sum(np.abs(S) ** 2, axis=1) - 1) < 5e-3)
    assert np.all(np.abs(S[:, 0, 0]) < 0.08)  # two 45-degree mitred bends
    beta = analytic.beta(W, H, ER, f)
    ph = -np.unwrap(np.angle(S[:, 1, 0]))
    L = ph / beta
    # mesh-converged: -0.06..-0.08 mm (mitred bends); a w/2.5 staircase grid
    # gives +0.9 mm on a 6 mm diagonal
    assert np.all(np.abs(L - (2 * mm + np.sqrt(2) * a)) < 0.12 * mm)


def test_tri_shunt_via_vs_openems():
    """test_vias.test_shunt_via_vs_openems on the unstructured mesh."""
    ER, H, W = 3.66, 0.254 * mm, 0.5561 * mm
    d = np.loadtxt(os.path.join(DATA, "via_shunt_openems.txt"))[[13, 39]]  # 3.75, 10.25 GHz
    f, s11_ref, s21_ref = d[:, 0], d[:, 1] + 1j * d[:, 2], d[:, 3] + 1j * d[:, 4]
    lay = Layout()
    lay.add_rect(-3 * mm, -W / 2, 3 * mm, W / 2)
    lay.add_port("P1", (-3 * mm, 0), "-x", W, 6 * mm)
    lay.add_port("P2", (3 * mm, 0), "+x", W, 6 * mm)
    lay.add_via(0, 0, drill=0.3 * mm, pad=W)
    m = lay.mesh(0.25 * mm, 0.05 * mm, kind="tri", min_angle=15)
    t = m.via_tris[0]
    assert t.size >= 3 and np.all(np.hypot(*m.centroid[t].T) < 0.15 * mm)
    S = Solver(m, Substrate(ER, H), verbose=False).sweep(f).renormalize(50)
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), np.abs(s21_ref), rtol=0.2)
    np.testing.assert_allclose(np.abs(S[:, 0, 0]), np.abs(s11_ref), atol=0.01)
    assert np.allclose(S[:, 0, 0], S[:, 1, 1], atol=3e-3)  # symmetric layout


def test_tri_oblique_ports():
    """A straight line at 30 degrees with feeds along it equals the same
    line along x (rotation invariance of the whole chain: mesh, cuts,
    voltage samples, wave fit)."""
    ER, H, W, L = 3.66, 0.254 * mm, 0.5561 * mm, 4 * mm
    f = np.array([3, 9]) * GHz
    out = []
    for ang in (0.0, 30.0):
        u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        from shapely.geometry import LineString

        lay = Layout()
        a, b = -0.5 * L * u, 0.5 * L * u
        lay.add_polygon(LineString([a, b]).buffer(W / 2, cap_style="flat"))
        lay.add_port("P1", tuple(a), ang + 180, W, 8 * mm)
        lay.add_port("P2", tuple(b), ang, W, 8 * mm)
        assert lay.ports[0].aligned == (ang == 0)
        m = lay.mesh(0.4 * mm, 0.06 * mm, kind="tri")
        out.append(Solver(m, Substrate(ER, H), verbose=False).sweep(f))
    r0, r1 = out
    np.testing.assert_allclose(r1.eeff, r0.eeff, rtol=2e-3)
    np.testing.assert_allclose(r1.z0_line.real, r0.z0_line.real, rtol=5e-3)
    np.testing.assert_allclose(r1.S, r0.S, atol=5e-3)
    with pytest.raises(ValueError):  # the tensor grid cannot hold an oblique feed
        lay.mesh(0.4 * mm, 0.06 * mm)
