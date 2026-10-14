import os

import numpy as np
import pytest
from scipy.special import ellipk

from mom25d import (GHz, Dielectric, GroundedSlabGreens, LayeredGreens, Layout, Solver, Stackup,
                    Substrate, mm)
from mom25d.constants import ETA0

DATA = os.path.join(os.path.dirname(__file__), "data")
R = np.array([1e-5, 1e-4, 5e-4, 2e-3, 1e-2, 0.03])


@pytest.mark.parametrize("f", [1e9, 10e9])
def test_layered_reduces_to_slab(f):
    g = GroundedSlabGreens(f, 3.55, 0.508e-3, 0.002, rho_max=0.04)
    p = LayeredGreens(f, Stackup.microstrip(3.55, 0.508e-3, 0.002), rho_max=0.04).pair("top", "top")
    sc = 4 * np.pi * R
    assert np.max(np.abs(p.gA(R) - g.gA(R)) * sc) < 1e-12
    assert np.max(np.abs(p.gV(R) - g.gV(R)) * sc) < 1e-12
    assert np.max(np.abs(p.gW(R) - g.gW(R)) * sc) < 1e-12


@pytest.mark.parametrize("f", [1e9, 20e9])
def test_layered_air_stack_matches_images(f):
    """Two air layers, metal on both interfaces: direct + negative image."""
    st = Stackup([Dielectric(0.3e-3, 1.0), Dielectric(0.5e-3, 1.0)], {"mid": 1, "top": 2})
    L = LayeredGreens(f, st, rho_max=0.04)
    for a in ("top", "mid"):
        for b in ("top", "mid"):
            za, zb = st.level_z(a), st.level_z(b)
            R1, R2 = np.hypot(R, za - zb), np.hypot(R, za + zb)
            ex = (np.exp(-1j * L.k0 * R1) / R1 - np.exp(-1j * L.k0 * R2) / R2) / (4 * np.pi)
            p = L.pair(a, b)
            assert np.max(np.abs(p.gA(R) - ex) * 4 * np.pi * R1) < 1e-6
            assert np.max(np.abs(p.gV(R) - ex) * 4 * np.pi * R1) < 1e-6


def test_split_substrate_equals_single_slab():
    """A substrate split into two identical layers must give the slab's S."""
    W = 0.5561 * mm
    lay = Layout()
    lay.add_rect(-3 * mm, -W / 2, 3 * mm, W / 2)
    lay.add_port("P1", (-3 * mm, 0), "-x", W, 6 * mm)
    lay.add_port("P2", (3 * mm, 0), "+x", W, 6 * mm)
    lay.add_via(0.7 * mm, 0, drill=0.3 * mm, pad=W)
    m = lay.mesh(max_cell=0.3 * mm, edge_cell=0.05 * mm)
    f = [3e9, 9e9]
    r1 = Solver(m, Substrate(3.66, 0.254 * mm, 0.003, 5.8e7), verbose=False).sweep(f)
    st = Stackup([Dielectric(0.1 * mm, 3.66, 0.003), Dielectric(0.154 * mm, 3.66, 0.003)],
                 {"top": 2}, sigma=5.8e7)
    s2 = Solver(m, st, verbose=False)
    assert not s2.single
    r2 = s2.sweep(f)
    assert np.max(np.abs(r1.S - r2.S)) < 1e-6


def test_stripline_vs_cohn():
    """Centred stripline under a PEC cover: eps_eff = eps_r, Z0 (zero
    thickness) = eta0 / (4 sqrt(eps_r)) K(k) / K(k'), k = sech(pi W / 2b)."""
    er, hh, W = 3.66, 0.254 * mm, 0.4 * mm
    b = 2 * hh
    k = 1 / np.cosh(np.pi * W / (2 * b))
    z0 = ETA0 / (4 * np.sqrt(er)) * ellipk(k**2) / ellipk(1 - k**2)
    lay = Layout()
    lay.add_rect(-2 * mm, -W / 2, 2 * mm, W / 2, layer="mid")
    lay.add_port("P1", (-2 * mm, 0), "-x", W, 6 * mm, layer="mid")
    lay.add_port("P2", (2 * mm, 0), "+x", W, 6 * mm, layer="mid")
    m = lay.mesh(max_cell=0.25 * mm, edge_cell=0.04 * mm)
    st = Stackup([Dielectric(hh, er), Dielectric(hh, er)], {"mid": 1}, top="pec")
    r = Solver(m, st, verbose=False).sweep([2e9, 10e9])
    np.testing.assert_allclose(r.eeff[:, 0], er, rtol=3e-3)
    np.testing.assert_allclose(r.z0_line[:, 0].real, z0, rtol=0.015)
    assert np.all(np.abs(r.S[:, 0, 0]) < 5e-3)


def test_via_segments_follow_the_stack():
    """A via from top to gnd through an inner layer with copper splits in
    two segments; without inner copper it is one."""
    st = Stackup([Dielectric(0.2 * mm, 3.66), Dielectric(0.2 * mm, 3.66)], {"inner": 1, "top": 2})
    for inner_copper, nseg in ((True, 2), (False, 1)):
        lay = Layout()
        lay.add_rect(-3 * mm, -0.3 * mm, 3 * mm, 0.3 * mm)
        lay.add_port("P1", (-3 * mm, 0), "-x", 0.6 * mm, 8 * mm)
        lay.add_port("P2", (3 * mm, 0), "+x", 0.6 * mm, 8 * mm)
        lay.add_rect(-4 * mm, -4 * mm, -3.5 * mm, -3.5 * mm, layer="inner")  # far away
        if inner_copper:
            lay.add_rect(-1 * mm, -1 * mm, 1 * mm, 1 * mm, layer="inner")
        lay.add_via(0, 0, 0.3 * mm, 0.6 * mm, top="top", bottom="gnd")
        m = lay.mesh(0.3 * mm)
        sol = Solver(m, st, verbose=False)
        assert m.n_vias == nseg
        assert m.via_segments[0]["top"] == "top" and m.via_segments[-1]["bottom"] == "gnd"
        L = sol.geometry().via_spans
        assert L[0][1] == pytest.approx(0.4 * mm) and L[-1][0] == 0.0


def test_top_to_inner_transition_vs_openems():
    """Top microstrip -> via -> buried line (two 0.254 mm layers) vs openEMS
    (tests/data/ref_multilayer_openems.py): S21 phase within 1.5 deg."""
    d = np.loadtxt(os.path.join(DATA, "transition_openems.txt"))
    d = d[[12, 36]]  # 3.5, 9.5 GHz
    f, s21_ref = d[:, 0], d[:, 3] + 1j * d[:, 4]
    WT, WI = 1.1122 * mm, 0.45 * mm
    lay = Layout()
    lay.add_rect(-3 * mm, -WT / 2, 0, WT / 2, layer="top")
    lay.add_rect(0, -WI / 2, 3 * mm, WI / 2, layer="inner")
    lay.add_port("P1", (-3 * mm, 0), "-x", WT, 12 * mm, layer="top")
    lay.add_port("P2", (3 * mm, 0), "+x", WI, 12 * mm, layer="inner")
    lay.add_via(0, 0, drill=0.3 * mm, pad=0.6 * mm, top="top", bottom="inner")
    m = lay.mesh(max_cell=0.3 * mm, edge_cell=0.05 * mm)
    st = Stackup([Dielectric(0.254 * mm, 3.66), Dielectric(0.254 * mm, 3.66)], {"inner": 1, "top": 2})
    S = Solver(m, st, verbose=False).sweep(f).renormalize(50)
    assert np.all(np.abs(np.angle(S[:, 1, 0] / s21_ref, deg=True)) < 1.5)
    assert np.all(np.abs(S[:, 0, 0]) < 0.06)
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), np.abs(s21_ref), atol=0.01)


def test_solder_mask_vs_openems():
    """A 25 um solder mask (eps_r 3.5, planar layer over everything) on a
    50-ohm microstrip, 20 mm: the S21 phase shift it causes vs openEMS
    (tests/data/ref_mask_openems.py) -- 11.6 vs 12.5 deg at 15 GHz."""
    idx = [36, 58]  # 9.5, 15 GHz
    ref = {}
    for c in ("bare", "mask"):
        d = np.loadtxt(os.path.join(DATA, f"mask_{c}_openems.txt"))[idx]
        ref[c] = d[:, 3] + 1j * d[:, 4]
    f = np.loadtxt(os.path.join(DATA, "mask_bare_openems.txt"))[idx, 0]
    W, H, L = 0.56 * mm, 0.254 * mm, 20 * mm
    lay = Layout()
    lay.add_rect(-L / 2, -W / 2, L / 2, W / 2)
    lay.add_port("P1", (-L / 2, 0), "-x", W, 12 * mm)
    lay.add_port("P2", (L / 2, 0), "+x", W, 12 * mm)
    m = lay.mesh(0.4 * mm, 0.05 * mm)
    S = {c: Solver(m, st, verbose=False).sweep(f, interp=None).renormalize(50)
         for c, st in (("bare", Stackup.microstrip(3.66, H)),
                       ("mask", Stackup([Dielectric(H, 3.66), Dielectric(0.025 * mm, 3.5)], {"top": 1})))}
    dm = np.angle(S["mask"][:, 1, 0] / S["bare"][:, 1, 0])
    do = np.angle(ref["mask"] / ref["bare"])
    np.testing.assert_allclose(dm, do, rtol=0.1)
    assert np.all(np.abs(S["mask"][:, 0, 0]) < 0.02)
