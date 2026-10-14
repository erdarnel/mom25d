import numpy as np
import pytest

from mom25d import Layout, Solver, Substrate, mm
from mom25d.network import connect, innerconnect, renormalize

ER, H, W = 3.66, 0.254 * mm, 0.56 * mm


def abcd(S, z0=50.0):
    S11, S12, S21, S22 = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    return np.array([[((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21), z0 * ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21)],
                     [((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21 * z0), ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)]])


def s_from_abcd(T, z0=50.0):
    A, B, C, D = T.ravel()
    den = A + B / z0 + C * z0 + D
    return np.array([[(A + B / z0 - C * z0 - D) / den, 2 * (A * D - B * C) / den],
                     [2 / den, (-A + B / z0 - C * z0 + D) / den]])


def attenuator_s2p(path, db=6.0):
    """Matched attenuator, ideal: S11 = S22 = 0, S21 = S12 = 10^(-db/20)."""
    a = 10 ** (-db / 20)
    path.write_text(f"# GHz S RI R 50\n0.1 0 0 {a} 0 {a} 0 0 0\n40 0 0 {a} 0 {a} 0 0 0\n")
    return str(path), a


def test_connect_matches_abcd_cascade():
    rng = np.random.default_rng(3)
    SA = 0.4 * (rng.standard_normal((1, 2, 2)) + 1j * rng.standard_normal((1, 2, 2)))
    SB = 0.4 * (rng.standard_normal((1, 2, 2)) + 1j * rng.standard_normal((1, 2, 2)))
    S, idx = connect(SA, SB, [(1, 0)])  # A port 2 -> B port 1
    assert idx == [0, 3]
    ref = s_from_abcd(abcd(SA[0]) @ abcd(SB[0]))
    np.testing.assert_allclose(S[0], ref, atol=1e-12)
    # joining both ports of a 2-port to a 2-port leaves nothing; a 3-port's two ports -> 1-port
    S3 = 0.3 * (rng.standard_normal((1, 3, 3)) + 1j * rng.standard_normal((1, 3, 3)))
    assert innerconnect(S3, 0, 2).shape == (1, 1, 1)


def test_renormalize_round_trip():
    rng = np.random.default_rng(4)
    S = 0.3 * (rng.standard_normal((2, 3, 3)) + 1j * rng.standard_normal((2, 3, 3)))
    z1, z2 = np.array([40.0, 50.0, 75.0]), 50.0
    np.testing.assert_allclose(renormalize(renormalize(S, z1, z2), z2, z1), S, atol=1e-12)


def line():
    lay = Layout()
    lay.add_rect(-5 * mm, -W / 2, 5 * mm, W / 2)
    lay.add_port("P1", (-5 * mm, 0), "-x", W, 10 * mm)
    lay.add_port("P2", (5 * mm, 0), "+x", W, 10 * mm)
    return lay


def test_sweep_connect_block(tmp_path):
    """A solved through line, its port P2 connected to a 6 dB attenuator
    file: S21 of the combination = S21(line, 50 ohm) x 0.5 (both ~matched),
    and equal to the exact ABCD cascade."""
    path, a = attenuator_s2p(tmp_path / "att6.s2p")
    f = np.array([2e9, 8e9])
    r = Solver(line().mesh(0.4 * mm, 0.05 * mm), Substrate(ER, H), verbose=False).sweep(f, interp=None)
    net = r.connect(path, [("P2", 1)])
    assert net.names == ["P1", "att6.2"] and net.z0 == 50.0
    S50 = r.renormalize(50.0)
    for k in range(len(f)):
        ref = s_from_abcd(abcd(S50[k]) @ abcd(np.array([[0, a], [a, 0]])))
        np.testing.assert_allclose(net.S[k], ref, atol=1e-10)
    np.testing.assert_allclose(np.abs(net.S[:, 1, 0]), np.abs(S50[:, 1, 0]) * a, atol=2e-3)
    net.write_touchstone(str(tmp_path / "out.s2p"))
    from mom25d.lumped import Touchstone

    np.testing.assert_allclose(Touchstone(str(tmp_path / "out.s2p")).s(f[0]), net.S[0], atol=1e-8)


def test_sparams_block_in_layout(tmp_path):
    """The same attenuator inside the layout between two line halves, its
    ports referenced to grounded pads (vias): S21 close to 0.5 at 1 GHz."""
    path, a = attenuator_s2p(tmp_path / "att6.s2p")
    lay = Layout()
    g = 0.6 * mm
    lay.add_rect(-5 * mm, -W / 2, -g, W / 2)
    lay.add_rect(g, -W / 2, 5 * mm, W / 2)
    lay.add_port("P1", (-5 * mm, 0), "-x", W, 10 * mm)
    lay.add_port("P2", (5 * mm, 0), "+x", W, 10 * mm)
    y = W / 2 + 0.4 * mm
    xa, xb = -g - 0.15 * mm, g + 0.15 * mm  # signal points on the two line ends
    lay.add_sparams("U1", path, [((xa, W / 2), (xa, y + 0.3 * mm)), ((xb, W / 2), (xb, y + 0.3 * mm))],
                    width=0.25 * mm, ground_pad=0.6 * mm)
    r = Solver(lay.mesh(0.3 * mm, 0.05 * mm), Substrate(ER, H), verbose=False).sweep([1e9], interp=None)
    assert [q.name for q in lay.lumped] == ["U1.1", "U1.2"]
    assert abs(r.renormalize(50)[0, 1, 0]) == pytest.approx(a, abs=0.06)


@pytest.mark.parametrize("load,s", [("open", 1.0), ("short", -1.0)])
def test_connected_block_sets_layout_currents(tmp_path, load, s):
    """P2 of a solved line connected to an open / short (.s1p): the layout
    currents recombined for a wave into P1 have no current (open) or no
    voltage (short) at P2's reference plane -- fitted from the standing
    wave on P2's feed: I(0) = A + B, V(0) ~ A - B."""
    p = tmp_path / f"{load}.s1p"
    p.write_text(f"# GHz S RI R 50\n0.1 {s} 0\n40 {s} 0\n")
    f = np.array([3e9])
    sol = Solver(line().mesh(0.4 * mm, 0.05 * mm), Substrate(ER, H), verbose=False)
    r = sol.sweep(f, keep_currents=True, interp=None)
    net = r.connect(str(p), [("P2", 1)])
    assert net.names == ["P1"] and abs(abs(net.S[0, 0, 0]) - 1) < 0.01  # lossless line, total reflection
    x = net.currents(0, "P1")
    pm = sol.pm[1]
    Ic = pm.currents(x[:, None])[:, 0]  # current toward the circuit at the feed cuts
    g = r.gamma[0, 1]
    M = np.c_[np.exp(g * pm.s), np.exp(-g * pm.s)]
    (A, B), *_ = np.linalg.lstsq(M, Ic, rcond=None)
    if load == "open":
        assert abs(A + B) < 0.02 * abs(A)
    else:
        assert abs(A - B) < 0.02 * abs(A)
    # a matched block reproduces the matched-port currents (line Z0 ~ 50 ohm)
    m = tmp_path / "match.s1p"
    m.write_text("# GHz S RI R 50\n0.1 0 0\n40 0 0\n")
    xm = r.connect(str(m), [("P2", 1)]).currents(0, "P1")
    x0 = r.wave_currents(0, [1.0, 0.0])
    np.testing.assert_allclose(xm, x0, atol=1e-12 * np.abs(x0).max())
