"""Regression tests for decisions 36-38 (2026-10-15): tri-mesh area bound, graded interiors, gap ports on a line
over a plane opening with port vias passing through the opening.   pytest -q Engine/tests/test_regressions_mesh_planes.py"""
import numpy as np
import pytest
from shapely.geometry import Point, box

from mom25d import Dielectric, GHz, Layout, Solver, Stackup, mm
from mom25d.trimesh import graded_size

W, G, BAND, L = 0.44, 0.41, 1.5, 1.5          # mm: line, gap, ground band, line length
H3, H1 = 0.130, 0.0754                         # mm: L3 -> L2, L2 -> L1


def _area(m):
    P = m.nodes[m.tris]
    return 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))


def test_tri_interior_area_bound():
    """Decision 37: the interior area bound reaches Triangle (it was passed in exponent notation and ignored)."""
    lay = Layout()
    lay.add_polygon(box(-3 * mm, -4 * mm, 3 * mm, 4 * mm), layer="top")
    mc = 0.15 * mm
    m = lay.mesh(mc, 0.04 * mm, kind="tri")
    assert _area(m).max() <= 0.5 * mc ** 2 * 1.01


def test_graded_interior():
    """Decision 38: size= grades the interior: fine near the given geometry, coarse (<= max_cell) far from it."""
    lay = Layout()
    lay.add_polygon(box(-5 * mm, -5 * mm, 5 * mm, 5 * mm), layer="top")
    fine = box(-0.2 * mm, -5 * mm, 0.2 * mm, 5 * mm)
    uni = lay.mesh(0.15 * mm, 0.04 * mm, kind="tri")
    gr = lay.mesh(0.8 * mm, 0.04 * mm, kind="tri", size=graded_size(fine, 0.15 * mm, 0.8 * mm, 0.3))
    a, c = _area(gr), gr.nodes[gr.tris].mean(1)
    near = np.abs(c[:, 0]) < 0.3 * mm
    assert a[near].max() <= 0.5 * (0.15 * mm + 0.3 * 0.3 * mm) ** 2 * 1.05
    assert a[~near].max() <= 0.5 * (0.8 * mm) ** 2 * 1.05
    assert len(gr.tris) < 0.5 * len(uni.tris)


def _stack():
    lam = [Dielectric(H3 * mm, 3.06, 0.004), Dielectric(H1 * mm, 3.06, 0.004), Dielectric(0.02 * mm, 3.5, 0.02)]
    return Stackup(lam, metals=dict(L1=2), planes=dict(L2=1), top="air")


def _line(void):
    """GCPW line on L1 over a plane L2 with `void` (mm polygon) open; gap ports to stitched pads beyond both ends,
    the pad vias reaching the ground through the opening when it extends under the pads."""
    lay = Layout()
    lay.add_polygon(box(-L / 2 * mm, -W / 2 * mm, L / 2 * mm, W / 2 * mm), layer="L1")
    for sy in (1, -1):
        lay.add_polygon(box(-L / 2 * mm, min(sy * (W / 2 + G), sy * (W / 2 + BAND)) * mm,
                            L / 2 * mm, max(sy * (W / 2 + G), sy * (W / 2 + BAND)) * mm), layer="L1")
    vias = [(x, sy * (W / 2 + G + 0.3)) for x in np.arange(-L / 2 + 0.2, L / 2 - 0.05, 0.4) for sy in (1, -1)]
    for sx in (1, -1):
        x0 = sx * (L / 2 + 0.15)
        x1 = sx * (L / 2 + 0.45)
        lay.add_polygon(box(min(x0, x1) * mm, -(W / 2 + 0.1) * mm, max(x0, x1) * mm, (W / 2 + 0.1) * mm), layer="L1")
        vias += [(sx * (L / 2 + 0.30), t) for t in (-0.12, 0.12)]
        lay.add_gap_port("P1" if sx < 0 else "P2", (sx * (L / 2 - 0.04) * mm, 0.0), (sx * (L / 2 + 0.19) * mm, 0.0),
                         W * mm, layer="L1", z_ref=50.0)
    for x, y in vias:
        con = () if void.contains(Point(x, y)) else ("L2",)
        lay.add_via(x * mm, y * mm, drill=0.152 * mm, pad=0.172 * mm, top="L1", bottom="gnd", connect=con)
    lay.add_plane("L2")
    from shapely import affinity
    lay.add_aperture("L2", affinity.scale(void, mm, mm, origin=(0, 0)), max_cell=0.15 * mm, edge_cell=0.04 * mm)
    return lay


def _solve(lay, f=29 * GHz):
    m = lay.mesh(0.2 * mm, 0.03 * mm, kind="tri")
    sol = Solver(m, _stack(), method="dense", verbose=False)
    return m, sol.sweep(np.array([f]), interp=None).renormalize(50.0)[0]


def test_gap_ports_over_plane_opening():
    """Decisions 36 + 37: a uniform line over an L2 opening (line + gaps wide, extending under the port pads, so the
    pad vias pass through it) is passive and symmetric. Before both fixes: 20-45 % missing, |S| > 1 near 33 GHz."""
    void = box(-L / 2 - 1.0, -(W / 2 + G + 0.05), L / 2 + 1.0, W / 2 + G + 0.05)
    m, S = _solve(_line(void))
    assert m.via_open_crossings().sum() == 4                      # the four pad vias pass through the opening
    lost = 1 - abs(S[0, 0]) ** 2 - abs(S[1, 0]) ** 2
    assert 0.0 <= lost < 0.12
    assert abs(abs(S[0, 0]) - abs(S[1, 1])) < 0.02
    assert abs(S[0, 1] - S[1, 0]) < 1e-4


@pytest.mark.xfail(reason="decision 37 open point: an opening over a whole model does not reproduce the model "
                          "without the plane", strict=False)
def test_full_opening_matches_no_plane():
    """An opening over all the copper (+1 mm) should give the line without L2 (same layout, L2 and L1 merged into
    one 205 um dielectric to L3)."""
    void = box(-L / 2 - 1.6, -(W / 2 + BAND + 1.0), L / 2 + 1.6, W / 2 + BAND + 1.0)
    _, S_open = _solve(_line(void))
    ref = _line(box(0, 0, 0, 0))                                   # no opening: rebuilt without the plane below
    ref.apertures, ref.planes = [], []
    for v in ref.vias:
        v.connect = ()
    lam = [Dielectric((H3 + H1) * mm, 3.06, 0.004), Dielectric(0.02 * mm, 3.5, 0.02)]
    m = ref.mesh(0.2 * mm, 0.03 * mm, kind="tri")
    S_ref = Solver(m, Stackup(lam, metals=dict(L1=1), top="air"), method="dense",
                   verbose=False).sweep(np.array([29 * GHz]), interp=None).renormalize(50.0)[0]
    assert np.abs(S_open - S_ref).max() < 0.05


def test_coincident_vias_between_planes():
    """Two vias at the same position in a closed (plane-to-plane) section gave rho = 0 and a division by zero in the
    TM_n sum; they are now treated like one barrel (distance = drill radius): finite and symmetric."""
    from types import SimpleNamespace
    from mom25d.solver import via_impedance
    vias = [SimpleNamespace(xy=(0.0, 0.0), drill=0.152e-3), SimpleNamespace(xy=(0.0, 0.0), drill=0.152e-3),
            SimpleNamespace(xy=(0.5e-3, 0.0), drill=0.152e-3)]
    spans = [(0.0, 0.13e-3)] * 3
    k = 2 * np.pi * 29e9 / 3e8 * np.sqrt(3.06)
    Z = via_impedance(vias, spans, 2 * np.pi * 29e9, [(0.0, 0.13e-3, k)])
    assert np.all(np.isfinite(Z)) and np.allclose(Z, Z.T)
    assert np.allclose(Z[0, 1], Z[0, 0])
