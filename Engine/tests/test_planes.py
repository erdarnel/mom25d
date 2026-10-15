import os

import numpy as np
import pytest

from mom25d import (Dielectric, LayeredGreens, Layout, SectionedGreens, Solver, Stackup, Via, mm,
                    via_impedance, via_inductance)
from mom25d.constants import MU0
from mom25d.solver import _filament_mutual_z

DATA = os.path.join(os.path.dirname(__file__), "data")
H = 0.254 * mm
R = np.array([1e-5, 1e-4, 5e-4, 2e-3, 1e-2])


def plane_stack(tand=0.0):
    """gnd | 0.254 | inner | 0.254 | L2 plane | 0.254 | top | air"""
    return Stackup([Dielectric(H, 3.66, tand)] * 3, {"inner": 1, "top": 3}, planes={"L2": 2})


def test_stackup_sections():
    st = plane_stack()
    assert st.bounds == [0, 2, 3]
    assert [st.section(n) for n in ("inner", "top")] == [0, 1]
    assert st.level_z("L2") == pytest.approx(2 * H)
    assert st.level_h("top") == pytest.approx(H) and st.level_h("inner") == pytest.approx(H)
    assert st.covered("inner") and not st.covered("top")
    assert st.section_stack(0).top == "pec" and st.section_stack(1).top == "air"
    with pytest.raises(ValueError):
        Stackup([Dielectric(H, 3.66)] * 2, {"top": 2}, planes={"L2": 2})  # plane on the top
    with pytest.raises(ValueError):
        Stackup([Dielectric(H, 3.66)] * 3, {"a": 1, "top": 3}, planes={"L2": 1})  # on metal
    with pytest.raises(ValueError):
        LayeredGreens(1e9, st)


def test_sectioned_greens_equal_separate_stacks():
    """Each section's kernels are those of its own stack; across the plane
    they vanish."""
    f = 10e9
    g = SectionedGreens(f, plane_stack(0.003), rho_max=0.03)
    ms = LayeredGreens(f, Stackup.microstrip(3.66, H, 0.003), rho_max=0.03).pair("top", "top")
    sl = LayeredGreens(f, Stackup([Dielectric(H, 3.66, 0.003)] * 2, {"inner": 1}, top="pec"),
                       rho_max=0.03).pair("inner", "inner")
    for name, ref in (("top", ms), ("inner", sl)):
        p = g.pair(name, name)
        for k in ("gA", "gV", "gW"):
            assert np.max(np.abs(getattr(p, k)(R) - getattr(ref, k)(R)) * 4 * np.pi * R) < 1e-12
    for a, b in (("top", "inner"), ("inner", "top")):
        p = g.pair(a, b)
        assert p.cA == p.cV == p.cW == 0 and not np.any(p.gV(R))


def test_parallel_plate_via_impedance():
    w = 2 * np.pi * 5e9
    vias = [Via((0, 0), 0.3 * mm), Via((1 * mm, 0.5 * mm), 0.2 * mm)]
    spans = [(0, 0.4 * mm), (0.1 * mm, 0.3 * mm)]
    # open section: the quasi-static via + image inductance
    Z = via_impedance(vias, spans, w, [(0, 0.4 * mm, None)])
    np.testing.assert_allclose(Z, 1j * w * via_inductance(vias, 0.4 * mm, spans=spans), rtol=1e-12)
    # via filling a parallel plate: radial line w mu d / 4 H0(2)(k a)
    from scipy.special import hankel2

    d, k = 0.5 * mm, w / 3e8 * np.sqrt(3.66)
    Z = via_impedance(vias[:1], [(0, d)], w, [(0, d, k)])
    assert Z[0, 0] == pytest.approx(w * MU0 * d / 4 * hankel2(0, k * 0.15 * mm), rel=1e-12)
    # partial piece, static limit: the TM_n series equals the (log-divergent)
    # image series up to a constant -> compare differences between radii
    a1, a2 = 0.1 * mm, 0.3 * mm
    m = np.arange(-100000, 100001)

    def images(r):
        tot = 0.0
        for s in (1, -1):
            b = np.sort(np.stack([2 * m * d + s * a1, 2 * m * d + s * a2]), axis=0)
            tot = tot + _filament_mutual_z(a1, a2, b[0], b[1], r).sum()
        return 1j * w * tot

    def modal(r):
        return via_impedance([Via((0, 0), 2 * r)], [(a1, a2)], w, [(0, d, 1e-6)])[0, 0]

    r1, r2 = 0.05 * mm, 0.2 * mm
    assert modal(r1) - modal(r2) == pytest.approx(images(r1) - images(r2), rel=1e-4)


def test_plane_layout_equals_separate_stacks():
    """A microstrip over L2 and a stripline under it, solved together, give
    the S of each solved alone and do not couple."""
    W = 0.4 * mm

    def layout(layers, plane=True):
        lay = Layout()
        if plane:
            lay.add_plane("L2")
        for n in layers:
            lay.add_rect(-2 * mm, -W / 2, 2 * mm, W / 2, layer=n)
            lay.add_port("P1" + n, (-2 * mm, 0), "-x", W, 6 * mm, layer=n)
            lay.add_port("P2" + n, (2 * mm, 0), "+x", W, 6 * mm, layer=n)
        return lay.mesh(max_cell=0.3 * mm, edge_cell=0.05 * mm)

    f = [3e9, 9e9]
    both = Solver(layout(["top", "inner"]), plane_stack(), verbose=False).sweep(f)
    top = Solver(layout(["top"], False), Stackup.microstrip(3.66, H), verbose=False).sweep(f)
    inner = Solver(layout(["inner"], False),
                   Stackup([Dielectric(H, 3.66)] * 2, {"inner": 1}, top="pec"), verbose=False).sweep(f)
    np.testing.assert_allclose(both.S[:, :2, :2], top.S, atol=1e-6)
    np.testing.assert_allclose(both.S[:, 2:, 2:], inner.S, atol=1e-6)
    assert np.max(np.abs(both.S[:, :2, 2:])) < 1e-9


def _transition(stitch):
    """Top microstrip -> via through L2 (antipad) -> stripline, optional
    stitching vias L2 -> gnd (tests/data/ref_plane_openems.py)."""
    WT, WI = 0.56 * mm, 0.24 * mm
    lay = Layout()
    lay.add_plane("L2")
    lay.add_rect(-3 * mm, -WT / 2, 0, WT / 2, layer="top")
    lay.add_rect(0, -WI / 2, 3 * mm, WI / 2, layer="inner")
    lay.add_port("P1", (-3 * mm, 0), "-x", WT, 12 * mm, layer="top")
    lay.add_port("P2", (3 * mm, 0), "+x", WI, 12 * mm, layer="inner")
    lay.add_via(0, 0, drill=0.3 * mm, pad=0.6 * mm, top="top", bottom="inner")
    for a in np.pi / 4 + np.arange(4 if stitch else 0) * np.pi / 2:
        lay.add_via(1 * mm * np.cos(a), 1 * mm * np.sin(a), drill=0.3 * mm, pad=0.6 * mm,
                    top="L2", bottom="gnd")
    return lay.mesh(max_cell=0.3 * mm, edge_cell=0.05 * mm)


def test_via_segments_with_planes():
    m = _transition(True)
    Solver(m, plane_stack(), verbose=False)
    segs = [(g["top"], g["bottom"]) for g in m.via_segments]
    assert segs == [("top", "inner")] + [("L2", "gnd")] * 4
    assert all(g["top_tris"] is None and g["bottom_tris"] is None for g in m.via_segments[1:])
    # a via from top to gnd connected to L2 has two segments
    lay = Layout()
    lay.add_plane("L2")
    lay.add_rect(-1 * mm, -1 * mm, 1 * mm, 1 * mm)
    lay.add_port("P1", (-1 * mm, 0), "-x", 0.5 * mm, 8 * mm)
    lay.add_via(0, 0, 0.3 * mm, 0.6 * mm, top="top", bottom="gnd", connect=("L2",))
    m = lay.mesh(0.3 * mm)
    Solver(m, plane_stack(), verbose=False)
    assert [(g["top"], g["bottom"]) for g in m.via_segments] == [("top", "L2"), ("L2", "gnd")]


@pytest.mark.parametrize("stitch", [False, True])
def test_plane_transition_vs_openems(stitch):
    """Top -> via through L2 -> stripline vs openEMS (ref_plane_openems.py).
    Lossless stack: without stitching, the lost power is the radial wave in
    the L2 / gnd parallel plate; 4 stitching vias suppress it. The MoM
    leak is ~25 % below openEMS (antipad aperture not modelled)."""
    d = np.loadtxt(os.path.join(DATA, ("plane_stitch" if stitch else "plane") + "_openems.txt"))
    d = d[[12, 44]]  # 3.5, 11.5 GHz
    f, s11, s21 = d[:, 0], d[:, 1] + 1j * d[:, 2], d[:, 3] + 1j * d[:, 4]
    r = Solver(_transition(stitch), plane_stack(), verbose=False).sweep(f)
    S = r.renormalize(50)
    np.testing.assert_allclose(r.eeff[:, 1], 3.66, rtol=2e-3)  # stripline
    assert np.all(np.abs(S[:, 0, 1] - S[:, 1, 0]) < 5e-3)
    assert np.all(np.abs(np.angle(S[:, 1, 0] / s21, deg=True)) < 1.0)
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), np.abs(s21), atol=0.015)
    lost = 1 - np.abs(S[:, 0, 0]) ** 2 - np.abs(S[:, 1, 0]) ** 2
    lost_ref = 1 - np.abs(s11) ** 2 - np.abs(s21) ** 2
    if stitch:
        assert np.all(lost < 0.01) and np.all(lost_ref < 0.01)
    else:
        assert np.all((lost > 0.6 * lost_ref) & (lost < 1.1 * lost_ref))
        assert lost[1] > 0.04


def test_cavity_leakage_power_balance():
    """Power the vias launch into the L2 / gnd cavity (leakage.py) carries the
    lost power of the lossless transition without stitching (the rest is
    radiation from the open top); per-via shares and the angular pattern add
    up to the total; four stitching vias suppress it."""
    from mom25d.leakage import cavity_leakage

    f = np.array([7.5e9, 15e9])
    for stitch in (False, True):
        sol = Solver(_transition(stitch), plane_stack(), verbose=False)
        r = sol.sweep(f, keep_currents=True)
        S = r.renormalize(50)
        for fi, fr in enumerate(f):
            x = r.wave_currents(fi, [1.0, 0.0])
            cav = cavity_leakage(sol, x, fr)
            assert len(cav) == 1 and cav[0].z == pytest.approx((0, 2 * H))
            c = cav[0]
            p = c.power / 0.5
            lost = 1 - np.sum(np.abs(S[fi][:, 0]) ** 2)
            phi = np.linspace(0, 2 * np.pi, 2001)[:-1]
            assert np.sum(c.pattern(phi)) * (phi[1] - phi[0]) == pytest.approx(c.power, rel=0.02, abs=5e-4)  # J0(k a) self terms: 0.1 % P_inc
            assert c.p_via.sum() == pytest.approx(c.power, rel=1e-12)
            if stitch:
                assert abs(p) < 1e-3
            else:
                assert 0.85 * lost < p < lost
