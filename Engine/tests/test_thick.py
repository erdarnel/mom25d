"""Sonnet-style thick metal (Layout.add_thickness): two sheets joined by
side-wall bases, against a quasi-static 2D finite-difference reference."""

import numpy as np
import pytest

from mom25d import Dielectric, Layout, Solver, Stackup, mm

H, ER, W, LINE, FEED = 0.508, 3.55, 1.1, 10.0, 12.0  # [mm]


def _line(t, kind="grid"):
    lay = Layout()
    lay.add_rect(0, -W / 2 * mm, LINE * mm, W / 2 * mm)
    lay.add_port("P1", (0, 0), "-x", W * mm, FEED * mm)
    lay.add_port("P2", (LINE * mm, 0), "+x", W * mm, FEED * mm)
    if t:
        lay.add_thickness("top", "top_t")
        st = Stackup([Dielectric(H * mm, ER), Dielectric(t * mm, 1.0)], {"top": 1, "top_t": 2})
    else:
        st = Stackup([Dielectric(H * mm, ER)], {"top": 1})
    return lay.mesh(0.4 * mm, edge_cell=0.08 * mm, kind=kind), st


def test_twin_mesh_and_walls():
    m, st = _line(0.035, "tri")
    assert m.layer_names == ["top", "top_t"]
    base = np.nonzero(m.twin_tri >= 0)[0]
    assert np.array_equal(m.verts[base], m.verts[m.twin_tri[base]])  # same triangles, same vertex order
    # one wall per boundary edge of the layer's copper
    tri = m.tris[base]
    e = np.sort(np.concatenate([tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [0, 1]]]), 1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert m.n_walls == np.sum(cnt == 1)
    w = m.wall
    assert np.all(m.tri_layer[m.tp[w]] == 0) and np.all(m.tri_layer[m.tm[w]] == 1)
    assert np.all(m.vp[w] == m.vm[w])
    # port cuts take both sheets, never walls
    p = m.layout.ports[0]
    s = m.cut_positions(p)
    idx, _ = m.cut_edges(p, s[s.size // 2])
    assert set(m.edge_layer[idx]) == {0, 1} and not m.wall[idx].any()


def test_thick_microstrip_vs_fd():
    """Copper thickness lowers Z0 and eps_eff. 2D FD reference (quasi-static,
    4 um grid) for t = 36 um: dZ0 -1.116 ohm, d eps_eff -0.0409 (Hammerstad-
    Jensen: -1.10 / -0.041 for 35 um). Two sheets give ~90 % of it (no
    charge on the side walls between the sheets, as in Sonnet's 2-sheet
    model): -0.91 / -0.033 at 5 GHz."""
    f = np.array([5e9])
    r0 = Solver(*_line(0.0), verbose=False).sweep(f, interp=None)
    r1 = Solver(*_line(0.036), verbose=False).sweep(f, interp=None)
    dz = r1.z0_line[0, 0].real - r0.z0_line[0, 0].real
    de = r1.eeff[0, 0] - r0.eeff[0, 0]
    assert -1.116 < dz < -0.75 * 1.116
    assert -0.0409 < de < -0.75 * 0.0409
    assert abs(r1.S[0, 0, 0]) < 0.01 and abs(abs(r1.S[0, 1, 0]) - 1) < 3e-3


def test_thick_hmatrix_matches_dense():
    m, st = _line(0.035, "tri")
    f = np.array([8e9])
    a = Solver(m, st, method="dense", verbose=False).sweep(f, interp=None)
    b = Solver(m, st, method="hmatrix", verbose=False).sweep(f, interp=None)
    assert np.abs(a.S - b.S).max() < 1e-4


def test_thick_loss_and_checks():
    m, st = _line(0.035)
    st.sigma, st.t = 5.8e7, 35e-6
    sol = Solver(m, st, verbose=False)
    zs, zv = sol._zs(1e6)
    assert np.allclose(zs.real, 2 / (5.8e7 * 35e-6), rtol=1e-3)  # each sheet: half the copper
    assert zv == pytest.approx(st.surface_impedance(1e6))
    # a twin level that is not next to the layer is refused
    bad = Stackup([Dielectric(H * mm, ER), Dielectric(0.02 * mm, 1.0), Dielectric(0.015 * mm, 1.0)],
                  {"top": 1, "mid": 2, "top_t": 3})
    with pytest.raises(ValueError, match="lie between"):
        Solver(m, bad, verbose=False)


def test_coincident_walls_two_levels():
    """Thick metal on two levels with strip edges on the same x-y line (a Manhattan grid puts
    both levels' walls at the same midpoints): the wall-wall mutual must stay finite (it was
    0 * inf = NaN, giving NaN port impedances)."""
    lay = Layout()
    lay.add_rect(0, -W / 2 * mm, LINE * mm, W / 2 * mm, layer="top")
    lay.add_rect(2 * mm, -W / 2 * mm, 8 * mm, W / 2 * mm, layer="mid")      # same edges in y
    lay.add_port("P1", (0, 0), "-x", W * mm, FEED * mm, layer="top")
    lay.add_port("P2", (LINE * mm, 0), "+x", W * mm, FEED * mm, layer="top")
    lay.add_thickness("top", "top_t")
    lay.add_thickness("mid", "mid_b")
    st = Stackup([Dielectric(0.2 * mm, ER), Dielectric(0.035 * mm, ER), Dielectric(0.2 * mm, ER), Dielectric(0.035 * mm, 1.0)],
                 {"mid_b": 1, "mid": 2, "top": 3, "top_t": 4})
    r = Solver(lay.mesh(0.4 * mm, edge_cell=0.08 * mm, kind="grid"), st, verbose=False).sweep(np.array([5e9]), interp=None)
    assert np.all(np.isfinite(r.S)) and np.all(np.isfinite(r.z0))
