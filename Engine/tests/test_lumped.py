import os

import numpy as np
import pytest
import scipy.linalg as sla

from mom25d import Layout, Solver, Substrate, mm
from mom25d.lumped import C, L, R, Touchstone, parallel, parse, series, series_from_s2p

DATA = os.path.join(os.path.dirname(__file__), "data")
ER, H, W = 3.66, 0.254 * mm, 0.56 * mm  # ~50 ohm microstrip
G = 0.5 * mm


def series_line(value, kind="grid", second=None):
    """50-ohm line, 5 mm either side of a 0.5 mm gap bridged by `value`;
    optional second gap 2 mm further on (for 2-port components)."""
    lay = Layout()
    lay.add_rect(-5 * mm, -W / 2, -G / 2, W / 2)
    lay.add_port("P1", (-5 * mm, 0), "-x", W, 10 * mm)
    lay.add_lumped("Z1", (-G / 2, 0), (G / 2, 0), W, value)
    if second is None:
        lay.add_rect(G / 2, -W / 2, 5 * mm, W / 2)
    else:
        x2 = 2 * mm
        lay.add_rect(G / 2, -W / 2, x2 - G / 2, W / 2)
        lay.add_lumped("Z2", (x2 - G / 2, 0), (x2 + G / 2, 0), W, second)
        lay.add_rect(x2 + G / 2, -W / 2, 5 * mm, W / 2)
    lay.add_port("P2", (5 * mm, 0), "+x", W, 10 * mm)
    return lay


def solver(lay, kind="grid"):
    return Solver(lay.mesh(0.4 * mm, 0.05 * mm, kind=kind), Substrate(ER, H), verbose=False)


def test_values(tmp_path):
    f = 1e9
    assert parse("R 50").z(f) == 50
    assert parse("C 1p").z(f) == pytest.approx(1 / (2j * np.pi * f * 1e-12))
    assert parse("R 10 + L 0.5n").z(f) == pytest.approx(10 + 2j * np.pi * f * 0.5e-9)
    assert parse("C 1p || R 1k").z(f) == pytest.approx(parallel(C(1e-12), R(1e3)).z(f))
    assert series(R(1), L(1e-9)).z(f) == pytest.approx(1 + 2j * np.pi * f * 1e-9)
    # Touchstone: 1-port of 25 ohm (S = -1/3), 2-port in v1 order S11 S21 S12 S22
    p1 = tmp_path / "r.s1p"
    p1.write_text("# GHz S RI R 50\n0.5 -0.333333333333 0\n20 -0.333333333333 0\n")
    assert Touchstone(str(p1)).z(3e9)[0, 0] == pytest.approx(25, rel=1e-9)
    p2 = tmp_path / "t.s2p"
    p2.write_text("# GHz S RI R 50\n1 0.1 0 0.2 0 0.3 0 0.4 0\n10 0.1 0 0.2 0 0.3 0 0.4 0\n")
    t = Touchstone(str(p2))
    s = np.array([[0.1, 0.3], [0.2, 0.4]])  # S21 = 0.2 is row 2, column 1
    assert t.z(5e9) == pytest.approx(50 * (np.eye(2) + s) @ np.linalg.inv(np.eye(2) - s))


@pytest.mark.parametrize("kind", ["grid", "tri"])
def test_series_resistor_matches_circuit(kind):
    """A series R across the gap: S21 = 2 Z0 / (2 Z0 + R), S11 = R / (2 Z0 + R)
    (referenced to the line's own Z0)."""
    r = solver(series_line(R(50)), kind).sweep([1e9, 3e9], interp=None)
    z0 = r.z0_line[:, 0].real
    np.testing.assert_allclose(np.abs(r.S[:, 1, 0]), 2 * z0 / (2 * z0 + 50), atol=1e-3)
    np.testing.assert_allclose(np.abs(r.S[:, 0, 0]), 50 / (2 * z0 + 50), atol=1e-3)


def test_woodbury_equals_direct_loading(tmp_path):
    """Loads applied after the solve (Woodbury) == loads added to Z, for
    one-port values and a 2-port component with coupling (full D)."""
    p2 = tmp_path / "c.s2p"
    p2.write_text("# GHz S RI R 50\n1 0.2 0.1 0.5 -0.2 0.5 -0.2 0.1 0.3\n10 0.2 0.1 0.5 -0.2 0.5 -0.2 0.1 0.3\n")
    lay = Layout()
    lay.add_rect(-5 * mm, -W / 2, -G / 2, W / 2)
    lay.add_port("P1", (-5 * mm, 0), "-x", W, 10 * mm)
    lay.add_lumped("A", (-G / 2, 0), (G / 2, 0), W)
    lay.add_rect(G / 2, -W / 2, 1.75 * mm, W / 2)
    lay.add_lumped("B", (1.75 * mm, 0), (2.25 * mm, 0), W)
    lay.add_rect(2.25 * mm, -W / 2, 5 * mm, W / 2)
    lay.add_port("P2", (5 * mm, 0), "+x", W, 10 * mm)
    lay.add_component(Touchstone(str(p2)), ["A", "B"])
    sol = solver(lay)
    f = 2e9
    Z = sol.impedance(f)
    E = np.stack([pm.excitation for pm in sol.pm], 1)
    x1 = sol.solve_currents(f, Z=Z.copy())
    D = sol.lumped_impedance(f)
    x2 = sla.solve(Z + sol.Lm @ D @ sol.Lm.T, E)
    assert np.abs(x1 - x2).max() < 1e-8 * np.abs(x2).max()


def test_value_override_and_open_gap():
    sol = solver(series_line(R(50)))
    f = [2e9]
    a = sol.sweep(f, interp=None, values={"Z1": R(100)}).S
    b = solver(series_line(R(100))).sweep(f, interp=None).S
    np.testing.assert_allclose(a, b, atol=1e-10)
    s_open = solver(series_line(None)).sweep(f, interp=None).S  # open gap: tiny coupling only
    assert abs(s_open[0, 1, 0]) < 0.05 and abs(s_open[0, 0, 0]) > 0.99


def test_shunt_resistor_to_ground():
    """A shunt R from the line to a grounded pad (via): S21 = 2 R / (2 R + Z0)."""
    lay = Layout()
    lay.add_rect(-5 * mm, -W / 2, 5 * mm, W / 2)
    lay.add_port("P1", (-5 * mm, 0), "-x", W, 10 * mm)
    lay.add_port("P2", (5 * mm, 0), "+x", W, 10 * mm)
    lay.add_lumped("R1", (0, W / 2), (0, W / 2 + G), 0.4 * mm, R(25))
    lay.add_rect(-0.4 * mm, W / 2 + G, 0.4 * mm, W / 2 + G + 0.8 * mm)  # ground pad
    lay.add_via(0, W / 2 + G + 0.4 * mm, drill=0.3 * mm, pad=0.6 * mm)
    r = solver(lay).sweep([1e9], interp=None)
    z0 = r.z0_line[0, 0].real
    assert abs(r.S[0, 1, 0]) == pytest.approx(2 * 25 / (2 * 25 + z0), abs=0.02)


def test_series_capacitor_vs_openems():
    """1 pF in the 0.5 mm gap vs an openEMS lumped capacitor
    (tests/data/ref_lumped_openems.py seriesC)."""
    d = np.loadtxt(os.path.join(DATA, "lumped_seriesC_openems.txt"))[[4, 20, 36, 52]]  # 1.5..13.5 GHz
    f, s11r, s21r = d[:, 0], d[:, 1] + 1j * d[:, 2], d[:, 3] + 1j * d[:, 4]
    S = solver(series_line(C(1e-12))).sweep(f, interp=None).renormalize(50)
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), np.abs(s21r), atol=0.03)
    np.testing.assert_allclose(np.abs(S[:, 0, 0]), np.abs(s11r), atol=0.05)
    assert np.all(np.abs(np.angle(S[:, 1, 0] / s21r, deg=True)) < 5)


def test_series_s2p_extrapolates(tmp_path):
    """Series-C .s2p generated from R + L + C over 1-10 GHz: the fit
    recovers the values and continues Z outside the band; in band = data."""
    r, l, c = 0.5, 0.15e-9, 0.2e-12
    f = np.linspace(1e9, 10e9, 40)
    z = r + 1j * (2 * np.pi * f * l - 1 / (2 * np.pi * f * c))
    s21 = 2 * 50 / (2 * 50 + z)
    s11 = z / (2 * 50 + z)
    p = tmp_path / "c.s2p"
    p.write_text("# Hz S RI R 50\n" + "".join(f"{fi} {a.real} {a.imag} {b.real} {b.imag} {b.real} {b.imag} {a.real} {a.imag}\n"
                                              for fi, a, b in zip(f, s11, s21)))
    v = parse(str(p))
    r0, r1, L, C = v.fit
    assert abs(L / l - 1) < 1e-6 and abs(C / c - 1) < 1e-6 and abs(r0 - r) < 1e-9
    w = 2 * np.pi * 20e9
    assert abs(v.z(20e9) - (r + 1j * (w * l - 1 / (w * c)))) < 1e-6
    z5 = r + 1j * (2 * np.pi * 5e9 * l - 1 / (2 * np.pi * 5e9 * c))
    assert abs(v.z(5e9) - z5) < 1e-3 * abs(z5)  # in band: interpolated data
    with pytest.raises(ValueError):
        series_from_s2p(str(p), extrapolate=False).z(20e9)
