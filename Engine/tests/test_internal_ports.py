"""Internal S-parameter ports: gap ports (Layout.add_gap_port) and via ports
at the ground plane (Layout.add_via_port); per-plane via antipads."""

import numpy as np
import pytest

from mom25d import GHz, Layout, Solver, Substrate, mm
from mom25d.lumped import R

from test_antipad import plane_stack

ER, H = 3.66, 0.254 * mm
W = 0.5561 * mm  # ~50 ohm
G = 0.5 * mm


def terminate(S, k, gamma):
    """S of the other ports with port k terminated in reflection gamma."""
    o = [i for i in range(S.shape[-1]) if i != k]
    return S[:, o][:, :, o] + S[:, o, k][:, :, None] * (gamma / (1 - S[:, k, k] * gamma))[:, None, None] \
        * S[:, k, o][:, None, :]


def series_gap(value=None, port=False):
    lay = Layout()
    lay.add_rect(-5 * mm, -W / 2, -G / 2, W / 2)
    lay.add_rect(G / 2, -W / 2, 5 * mm, W / 2)
    lay.add_port("P1", (-5 * mm, 0), "-x", W, 10 * mm)
    lay.add_port("P2", (5 * mm, 0), "+x", W, 10 * mm)
    if port:
        lay.add_gap_port("G", (-G / 2, 0), (G / 2, 0), W)
    else:
        lay.add_lumped("Z", (-G / 2, 0), (G / 2, 0), W, value)
    return lay


def shunt_via(port=True):
    lay = Layout()
    lay.add_rect(-3 * mm, -W / 2, 3 * mm, W / 2)
    lay.add_port("P1", (-3 * mm, 0), "-x", W, 6 * mm)
    lay.add_port("P2", (3 * mm, 0), "+x", W, 6 * mm)
    lay.add_via(0, 0, drill=0.3 * mm, pad=W)
    if port:
        lay.add_via_port("V", 0)
    return lay


def solve(lay, f, cell=0.4 * mm, z0=(50.0, 50.0)):
    s = Solver(lay.mesh(cell, 0.05 * mm), Substrate(ER, H), verbose=False, z0=None if z0 is None else list(z0))
    return s.sweep(f, interp=None)


def reciprocal(S, atol=3e-3):
    return np.allclose(S, np.swapaxes(S, 1, 2), atol=atol)


def test_gap_port_matches_lumped_loads():
    """A gap port terminated in r reproduces the lumped R(r) solve; a series
    50 ohm at low f gives S21 = 2/3; reciprocal."""
    f = np.array([1, 8]) * GHz
    assert reciprocal(solve(series_gap(port=True), f, z0=None).S)  # feed Z0 anchored to the gap port
    res = solve(series_gap(port=True), f)  # fixed 50 ohm feed references: comparable with the lumped solve
    assert res.port_names == ["P1", "P2", "G"]
    assert np.allclose(res.z0[:, 2], 50.0)
    S = res.S
    for r in (10.0, 200.0):
        ref = solve(series_gap(R(r)), f).S
        got = terminate(S, 2, np.full(len(f), (r - 50) / (r + 50)))
        assert np.allclose(got, ref, atol=2e-3)
    m = terminate(S, 2, np.zeros(len(f)))
    assert abs(m[0, 1, 0]) == pytest.approx(2 / 3, abs=0.03)


def test_via_port():
    """Via fed through the ground plane at the end of a shunt via: shorted it
    is the plain shunt via; 50 ohm at 3 GHz is about a shunt 50 ohm (S21 = 2/3,
    which checks the gap normalisation); open it is a through line. (At
    1 GHz the plain shunt via reflects almost totally and its wave fit
    alone is poorly conditioned: S11 differs by 0.02 there.)"""
    f = np.array([3, 6]) * GHz
    Sr = solve(shunt_via(), f, cell=0.25 * mm, z0=None).S
    assert reciprocal(Sr)
    assert np.all(np.abs(np.sum(np.abs(Sr) ** 2, axis=1) - 1) < 0.02)  # lossless
    res = solve(shunt_via(), f, cell=0.25 * mm)
    assert res.port_names == ["P1", "P2", "V"]
    S = res.S
    ref = solve(shunt_via(port=False), f, cell=0.25 * mm).S
    assert np.allclose(terminate(S, 2, -np.ones(len(f))), ref, atol=3e-3)
    m = terminate(S, 2, np.zeros(len(f)))
    assert abs(m[0, 1, 0]) == pytest.approx(2 / 3, abs=0.03)
    assert abs(m[0, 0, 0]) == pytest.approx(1 / 3, abs=0.03)  # phase: 2 x 3 mm of line
    assert np.all(np.abs(terminate(S, 2, np.ones(len(f)))[:, 1, 0]) > 0.97)


def test_internal_ports_only():
    """No feed ports: a gap port and a via port at the two ends of a line."""
    lay = Layout()
    lay.add_rect(-3 * mm, -W / 2, 3 * mm, W / 2)
    lay.add_rect(-3.6 * mm, -W / 2, -3.4 * mm, W / 2)  # grounded pad next to the line end
    lay.add_via(-3.5 * mm, 0, drill=0.15 * mm, pad=0.2 * mm)
    lay.add_gap_port("A", (-3 * mm + 0.1 * mm, 0), (-3.5 * mm, 0), 0.2 * mm)
    lay.add_via(2.9 * mm, 0, drill=0.15 * mm, pad=0.2 * mm)
    lay.add_via_port("B", 1)
    res = Solver(lay.mesh(0.25 * mm, 0.05 * mm), Substrate(ER, H), verbose=False).sweep([2 * GHz], interp=None)
    S = res.S[0]
    assert res.port_names == ["A", "B"]
    assert S[0, 1] == pytest.approx(S[1, 0], abs=3e-3)
    assert abs(S[1, 0]) > 0.7


def test_per_plane_antipad():
    """antipad={plane: d}: a frill only in the listed planes, with that d."""
    from test_antipad import transition

    m = transition({"L2": 0.7 * mm})
    Solver(m, plane_stack(), verbose=False)
    assert len(m.frills) == 1 and m.frills[0]["b"] == pytest.approx(0.35 * mm)
    m = transition({"other": 0.7 * mm})
    Solver(m, plane_stack(), verbose=False)
    assert len(m.frills) == 0
    with pytest.raises(ValueError):
        transition({"L2": 0.2 * mm})


def _line_with_pad_ports(bridge_in):
    """Line with a grounded pad beyond each end and gap ports line end -> pad; bridge_in: how far each bridge end
    reaches into the copper (0: the ends only touch the copper edges)."""
    lay = Layout()
    lay.add_rect(-3 * mm, -W / 2, 3 * mm, W / 2)
    for sx, name in ((-1, "A"), (1, "B")):
        xe, xp0, xp1 = sx * 3 * mm, sx * 3.3 * mm, sx * 3.6 * mm
        lay.add_rect(min(xp0, xp1), -0.15 * mm, max(xp0, xp1), 0.15 * mm)
        lay.add_via(sx * 3.45 * mm, 0, drill=0.15 * mm, pad=0.2 * mm)
        lay.add_gap_port(name, (xe - sx * bridge_in, 0), (xp0 + sx * bridge_in, 0), 0.2 * mm)
    return lay


def test_tand_floor_with_internal_ports():
    """Opt-in tand_floor: with internal ports and every dielectric lossless the solver applies it and warns; the
    result equals an explicit tan d 1e-5 run; the default (None) keeps tan d = 0."""
    f = [10 * GHz]
    lay = _line_with_pad_ports(0.04 * mm)
    m = lay.mesh(0.25 * mm, 0.05 * mm)
    with pytest.warns(UserWarning, match="tand_floor"):
        s0 = Solver(m, Substrate(ER, H), verbose=False, tand_floor=1e-5)
    assert all(d.tand == pytest.approx(1e-5) for d in s0.stack.layers) and s0.lossless
    S0 = s0.sweep(f, interp=None).S
    S1 = Solver(m, Substrate(ER, H, tand=1e-5), verbose=False).sweep(f, interp=None).S
    assert np.allclose(S0, S1, atol=1e-6)
    s2 = Solver(m, Substrate(ER, H), verbose=False)
    assert all(d.tand == 0 for d in s2.stack.layers)
