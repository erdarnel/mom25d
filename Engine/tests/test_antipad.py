import os

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

import mom25d.mesh as M
from mom25d import Dielectric, Layout, Solver, Stackup, mm
from mom25d.antipad import FrillModel, ring_nodes, ring_spectrum, ring_weight

DATA = os.path.join(os.path.dirname(__file__), "data")
H = 0.254 * mm


def plane_stack():
    return Stackup([Dielectric(H, 3.66)] * 3, {"inner": 1, "top": 3}, planes={"L2": 2})


def transition(antipad, stitch=True):
    """tests/data/ref_plane_openems.py: top microstrip -> via through L2 ->
    stripline, 4 stitching vias L2 -> gnd."""
    WT, WI = 0.56 * mm, 0.24 * mm
    lay = Layout()
    lay.add_plane("L2")
    lay.add_rect(-3 * mm, -WT / 2, 0, WT / 2, layer="top")
    lay.add_rect(0, -WI / 2, 3 * mm, WI / 2, layer="inner")
    lay.add_port("P1", (-3 * mm, 0), "-x", WT, 12 * mm, layer="top")
    lay.add_port("P2", (3 * mm, 0), "+x", WI, 12 * mm, layer="inner")
    lay.add_via(0, 0, drill=0.3 * mm, pad=0.6 * mm, top="top", bottom="inner", antipad=antipad)
    for a in np.pi / 4 + np.arange(4 if stitch else 0) * np.pi / 2:
        lay.add_via(1 * mm * np.cos(a), 1 * mm * np.sin(a), drill=0.3 * mm, pad=0.6 * mm,
                    top="L2", bottom="gnd")
    return lay.mesh(max_cell=0.3 * mm, edge_cell=0.05 * mm)


def test_ring_spectrum():
    """Exact Hankel transforms of the ring functions (Gauss rule at small k,
    closed form with Struve functions on the real axis beyond)."""
    a, b = 0.15 * mm, 0.35 * mm
    nodes = ring_nodes(a, b, 4)
    for j in (0, 2, 3):
        for k in (1e3, 3e4, 2e5):
            ex = quad(lambda r: 2 * np.pi * ring_weight(j, r, nodes) * jv(0, k * r) * r, 0, b,
                      limit=2000, points=list(nodes))[0]
            assert ring_spectrum(j, np.array([k + 0j]), nodes)[0].real == pytest.approx(ex, rel=1e-6, abs=1e-18)


def test_via_split_and_unknowns():
    m = transition(0.7 * mm)
    Solver(m, plane_stack(), verbose=False)
    assert len(m.frills) == 1 and m.n_frills == M.FRILL_RINGS
    fr = m.frills[0]
    up, low = m.via_segments[fr["seg_up"]], m.via_segments[fr["seg_low"]]
    assert (up["top"], up["bottom"], low["top"], low["bottom"]) == ("top", "L2", "L2", "inner")
    assert up["bottom_tris"] is None and low["top_tris"] is None  # no charge at the plane
    assert m.n_unknowns == m.n_basis + m.n_vias + m.n_frills
    # without antipad= the via passes the plane as one segment
    m0 = transition(None)
    Solver(m0, plane_stack(), verbose=False)
    assert m0.n_frills == 0 and m0.n_vias == m.n_vias - 1


def test_frill_capacitance_converges_and_falls_with_hole_size():
    """Barrel-to-plane capacitance (other rings eliminated): converged to a
    few % at 4 rings, smaller for larger holes."""
    out = {}
    for nr in (4, 8):
        M.FRILL_RINGS = nr
        try:
            for ap in (0.4, 0.7, 1.2):
                sol = Solver(transition(ap * mm), plane_stack(), verbose=False)
                g = sol.greens(2e9)
                _, F = FrillModel(sol).blocks(g, g.omega)
                C = (F / (1j * g.omega)).real
                out[nr, ap] = C[0, 0] - C[0, 1:] @ np.linalg.solve(C[1:, 1:], C[1:, 0])
        finally:
            M.FRILL_RINGS = 4
    for ap in (0.4, 0.7, 1.2):
        assert out[4, ap] == pytest.approx(out[8, ap], rel=0.05)
    assert out[8, 0.4] > out[8, 0.7] > out[8, 1.2] > 0


def test_antipad_trend_vs_openems():
    """Shrinking / growing the antipad (0.7 -> 0.4 / 1.2 mm) moves |S11| and
    the S21 phase the same way as openEMS (refined FDTD mesh, fine 20 um).
    S21 phase change at 15 GHz: 0.7 -> 0.4 mm model -5.4 vs -6.5 deg (the
    normal FDTD mesh gave -8.7: one cell across the 50 um gap); 0.7 -> 1.2
    mm +6.5 vs +4.1 deg and |S11| 0.17 vs 0.05 -- large holes overshoot:
    only the axisymmetric aperture field is modelled."""
    idx = [36, 58]  # 9.5, 15 GHz
    ref = {}
    for ap, name in ((0.4, "plane_stitch_anti400_fine20"), (0.7, "plane_stitch_anti700_fine20"),
                     (1.2, "plane_stitch_anti1200_fine20")):
        d = np.loadtxt(os.path.join(DATA, name + "_openems.txt"))[idx]
        ref[ap] = (d[:, 1] + 1j * d[:, 2], d[:, 3] + 1j * d[:, 4])
    f = np.loadtxt(os.path.join(DATA, "plane_stitch_openems.txt"))[idx, 0]
    S = {ap: Solver(transition(ap * mm), plane_stack(), verbose=False).sweep(f, interp=None).renormalize(50)
         for ap in (0.4, 0.7, 1.2)}
    for s in S.values():
        assert np.abs(s[:, 0, 1] - s[:, 1, 0]).max() < 5e-3  # reciprocal
        assert np.all(np.sum(np.abs(s) ** 2, axis=1) < 1 + 2e-3)  # passive
    assert np.all(np.abs(S[0.4][:, 0, 0]) > np.abs(S[0.7][:, 0, 0]))
    assert np.all(np.abs(S[0.7][:, 0, 0]) > np.abs(S[1.2][:, 0, 0]))
    for ap, lo, hi in ((0.4, 0.7, 1.2), (1.2, 1.0, 2.0)):
        dm = np.angle(S[ap][:, 1, 0] / S[0.7][:, 1, 0])
        do = np.angle(ref[ap][1] / ref[0.7][1])
        assert np.all(np.sign(dm) == np.sign(do))
        assert np.all((np.abs(dm) > lo * np.abs(do)) & (np.abs(dm) < hi * np.abs(do)))
    np.testing.assert_allclose(np.abs(S[0.4][:, 0, 0]), np.abs(ref[0.4][0]), atol=0.07)
    # absolute, 0.7 mm: |S21| and phase close to openEMS
    np.testing.assert_allclose(np.abs(S[0.7][:, 1, 0]), np.abs(ref[0.7][1]), atol=0.025)
    assert np.all(np.abs(np.angle(S[0.7][:, 1, 0] / ref[0.7][1], deg=True)) < 4)
