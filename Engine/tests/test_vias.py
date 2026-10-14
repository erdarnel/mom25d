import os

import numpy as np
import pytest

from mom25d import GHz, Layout, Solver, Substrate, mm, via_inductance
from mom25d.solver import _filament_mutual

ER, H = 3.66, 0.254 * mm
W = 0.5561 * mm  # ~50 ohm
DATA = os.path.join(os.path.dirname(__file__), "data")


def shunt_via_line(drill=0.3 * mm, cell=0.25 * mm):
    lay = Layout()
    lay.add_rect(-3 * mm, -W / 2, 3 * mm, W / 2)
    lay.add_port("P1", (-3 * mm, 0), "-x", W, 6 * mm)
    lay.add_port("P2", (3 * mm, 0), "+x", W, 6 * mm)
    lay.add_via(0, 0, drill=drill, pad=W)
    return lay.mesh(max_cell=cell, edge_cell=0.05 * mm)


def test_filament_inductance_limits():
    """Partial self inductance of a long thin wire -> mu0 l/2pi (ln(2l/a) - 1)."""
    l, a = 1.0, 1e-4
    ref = 2e-7 * l * (np.log(2 * l / a) - 1)
    assert _filament_mutual(l, a) == pytest.approx(ref, rel=2e-4)  # + a/l term
    # far apart: M -> mu0 l^2 / (4 pi d)
    assert _filament_mutual(1e-3, 1.0) == pytest.approx(1e-7 * 1e-6 / 1.0, rel=1e-3)


def test_via_mesh_attachment():
    m = shunt_via_line()
    assert m.n_vias == 1 and m.n_unknowns == m.n_basis + 1
    t = m.via_tris[0]
    assert t.size > 0
    assert np.all(np.hypot(*m.centroid[t].T) < W / 2)


def test_shunt_via_line_physics():
    """Lossless line with a via to ground in the middle: reciprocal, nearly
    unitary, a near-short at low f, and a thinner via (more inductance) lets
    more through."""
    f = np.array([2, 10]) * GHz
    s21 = []
    for drill in (0.3 * mm, 0.1 * mm):
        res = Solver(shunt_via_line(drill), Substrate(ER, H), verbose=False).sweep(f)
        S = res.renormalize(50)
        assert np.allclose(S[:, 0, 1], S[:, 1, 0], atol=2e-3)
        assert np.allclose(S[:, 0, 0], S[:, 1, 1], atol=2e-3)
        assert np.all(np.abs(np.sum(np.abs(S) ** 2, axis=1) - 1) < 0.015)
        assert np.all(np.abs(S[:, 0, 0]) > 0.95)
        s21.append(np.abs(S[:, 1, 0]))
    assert np.all(s21[1] > 1.5 * s21[0])
    assert np.all(s21[0][1] > 1.5 * s21[0][0])  # inductive: grows with f


def test_pour_clearance_and_fence():
    shapely = pytest.importorskip("shapely")
    lay = Layout()
    lay.add_rect(-3 * mm, -W / 2, 3 * mm, W / 2)
    lay.add_port("P1", (-3 * mm, 0), "-x", W, 4 * mm)
    lay.add_port("P2", (3 * mm, 0), "+x", W, 4 * mm)
    G = 0.2 * mm
    lay.add_pour((-7 * mm, -2 * mm, 7 * mm, 2 * mm), G, via_pitch=1 * mm, via_drill=0.3 * mm,
                 via_pad=0.5 * mm)
    (geom, vias), = lay.resolved_pours()
    strip = shapely.geometry.box(-7 * mm, -W / 2, 7 * mm, W / 2)
    assert geom.distance(strip) == pytest.approx(G, rel=1e-6)
    assert len(geom.geoms) == 2
    assert len(vias) > 20
    for v in vias:  # pads fully inside the pour
        assert geom.contains(shapely.geometry.Point(v.xy).buffer(v.pad / 2 * 0.99))
    m = lay.mesh(0.4 * mm)
    assert m.n_vias == len(vias)
    # pour edges are grid lines -> gap exactly G
    assert np.any(np.isclose(m.yg, W / 2 + G))


def test_via_inductance_matrix():
    from mom25d import Via

    vs = [Via((0, 0), 0.3 * mm), Via((1 * mm, 0), 0.3 * mm)]
    L = via_inductance(vs, H)
    assert L.shape == (2, 2) and np.allclose(L, L.T)
    assert L[0, 0] > L[0, 1] > 0


def test_shunt_via_vs_openems():
    """Through line with a 0.3 mm via to ground in the middle vs openEMS
    (tests/data/ref_vias_openems.py 'via'), referenced 3 mm either side."""
    d = np.loadtxt(os.path.join(DATA, "via_shunt_openems.txt"))
    d = d[[13, 26, 39, 52]]  # 3.75, 7, 10.25, 13.5 GHz
    f, s21_ref = d[:, 0], d[:, 3] + 1j * d[:, 4]
    s11_ref = d[:, 1] + 1j * d[:, 2]
    res = Solver(shunt_via_line(cell=0.15 * mm), Substrate(ER, H), verbose=False).sweep(f)
    S = res.renormalize(50)
    # |S21| ~ omega L_via / Z0: effective via inductance within 20 %
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), np.abs(s21_ref), rtol=0.2)
    np.testing.assert_allclose(np.abs(S[:, 0, 0]), np.abs(s11_ref), atol=0.01)
    ang = np.angle(S[:, 0, 0] / s11_ref, deg=True)
    assert np.all(np.abs(ang) < 5)  # ~70 um reference-plane offset (FDTD mesh snapping)
