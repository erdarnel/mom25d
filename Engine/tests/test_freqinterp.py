import numpy as np
import pytest

from mom25d import FreqInterp, GHz, Layout, Solver, Substrate, mm


def notch_solver():
    W, Ls = 0.6 * mm, 12 * mm
    lay = Layout()
    lay.add_rect(-W / 2 - 1 * mm, -W / 2, W / 2 + 1 * mm, W / 2)
    lay.add_rect(-W / 2, W / 2, W / 2, W / 2 + Ls)
    lay.add_port("P1", (-W / 2 - 1 * mm, 0), "-x", W, 12 * mm)
    lay.add_port("P2", (W / 2 + 1 * mm, 0), "+x", W, 12 * mm)
    return Solver(lay.mesh(0.5 * mm, 0.05 * mm), Substrate(3.66, 0.254 * mm, 0.003, 5.8e7),
                  verbose=False)


def test_lagrange_weights_and_window():
    fi = FreqInterp(step=1e9, order=3)
    fi.anchors = np.arange(1, 10) * 1e9
    for f in (1e9, 2.3e9, 5.5e9, 9e9):
        idx = fi._window(f)
        assert len(idx) == 4 and idx == sorted(idx)
        assert fi.anchors[idx[0]] <= f <= fi.anchors[idx[-1]]
        L = fi.weights(f, idx)
        assert L.sum() == pytest.approx(1)
        assert L @ fi.anchors[idx] == pytest.approx(f)  # exact for polynomials <= order


def test_interpolated_sweep_matches_exact():
    """Notch filter (stub resonance in band), 1-9 GHz: cubic interpolation
    from 4 anchor fills (1, 3.67, 6.33, 9 GHz) reproduces the exact sweep to
    |dS| < 1e-3; at the anchors it is exact."""
    sol = notch_solver()
    f = np.linspace(1, 9, 17) * GHz
    ex = sol.sweep(f, interp=None).renormalize(50)
    fi = FreqInterp(step=4e9, order=3)
    S = sol.sweep(f, interp=fi).renormalize(50)
    assert fi.n_fills == 4
    assert np.abs(S - ex).max() < 1e-3
    on = [0, -1]  # requested points that coincide with anchors (1, 9 GHz)
    assert np.abs(S[on] - ex[on]).max() < 1e-5
    # the anchor cache never holds more than order + 1 matrices
    assert len(fi._W) <= fi.order + 1


def test_interp_error_report():
    sol = notch_solver()
    fi = FreqInterp(step=2e9, order=2).setup(sol, np.array([2e9, 6e9]))
    assert fi.report_error(3e9) < 1e-3


def test_interp_with_vias():
    """Several grounding vias: the via-via block (large, distance-independent
    Im gV(0) correction) must not get the phase extraction -- with it, that
    block interpolated to 0.6-6 %; every block must be accurate on its own."""
    W = 0.5561 * mm
    lay = Layout()
    lay.add_rect(-3 * mm, -W / 2, 3 * mm, W / 2)
    lay.add_port("P1", (-3 * mm, 0), "-x", W, 6 * mm)
    lay.add_port("P2", (3 * mm, 0), "+x", W, 6 * mm)
    for x in (-1.5, 0.0, 1.5):
        lay.add_via(x * mm, 0, drill=0.3 * mm, pad=W)
    sol = Solver(lay.mesh(0.3 * mm, 0.05 * mm), Substrate(3.66, 0.254 * mm, 0.0037, 5.8e7),
                 verbose=False)
    fi = FreqInterp(step=1.5e9, order=3).setup(sol, np.array([1e9, 5.5e9]))
    assert fi.report_error(1.6e9) < 1e-5
    assert fi.report_error(4.2e9) < 1e-5


def test_auto_interp_default():
    """sweep() interpolates by default only when it saves fills."""
    sol = notch_solver()
    few = sol.sweep(np.array([2, 4]) * GHz)  # 2 points: exact
    many = sol.sweep(np.linspace(2, 4, 9) * GHz)  # 9 points, 4 anchors: interpolated
    np.testing.assert_allclose(many.S[[0, -1]], few.S, atol=1e-6)  # band edges are anchors
