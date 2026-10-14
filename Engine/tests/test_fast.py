"""Compiled (numba) fill kernels == the numpy path (same Z, dense and
H-matrix, single layer + vias + loss, layered with an inner via)."""

import numpy as np
import pytest

from mom25d import Dielectric, Layout, Solver, Stackup, Substrate, _fast, mm
from mom25d.hmatrix import HOperator

pytestmark = pytest.mark.skipif(not _fast.ENABLED, reason="numba not installed / disabled")

W = 0.5561 * mm


def gcpw():
    lay = Layout()
    lay.add_rect(-2 * mm, -W / 2, 2 * mm, W / 2)
    lay.add_port("P1", (-2 * mm, 0), "-x", W, 4 * mm)
    lay.add_port("P2", (2 * mm, 0), "+x", W, 4 * mm)
    y1 = W / 2 + 1.2 * mm
    lay.add_pour((-6 * mm, -y1, 6 * mm, y1), 0.2 * mm, via_pitch=1 * mm, via_drill=0.3 * mm, via_pad=0.5 * mm)
    return lay.mesh(0.4 * mm, 0.08 * mm, kind="tri")


def transition():
    WT, WI = 1.1122 * mm, 0.45 * mm
    lay = Layout()
    lay.add_rect(-2 * mm, -WT / 2, 0, WT / 2, layer="top")
    lay.add_rect(0, -WI / 2, 2 * mm, WI / 2, layer="inner")
    lay.add_port("P1", (-2 * mm, 0), "-x", WT, 8 * mm, layer="top")
    lay.add_port("P2", (2 * mm, 0), "+x", WI, 8 * mm, layer="inner")
    lay.add_via(0, 0, drill=0.3 * mm, pad=0.6 * mm, top="top", bottom="inner")
    st = Stackup([Dielectric(0.254 * mm, 3.66), Dielectric(0.254 * mm, 3.66)], {"inner": 1, "top": 2})
    return lay.mesh(max_cell=0.4 * mm, edge_cell=0.08 * mm), st


def both(fn):
    old = _fast.ENABLED
    try:
        _fast.ENABLED = False
        a = fn()
        _fast.ENABLED = True
        return a, fn()
    finally:
        _fast.ENABLED = old


@pytest.mark.parametrize("case", ["gcpw", "layered"])
def test_dense_fill_matches_numpy(case):
    m, st = (gcpw(), Substrate(3.66, 0.254 * mm, 0.0037, 5.8e7)) if case == "gcpw" else transition()
    Z0, Z1 = both(lambda: Solver(m, st, verbose=False, method="dense").impedance(7e9))
    assert np.abs(Z1 - Z0).max() < 1e-12 * np.abs(Z0).max()


def test_hmatrix_fill_matches_numpy():
    m = gcpw()
    sol = Solver(m, Substrate(3.66, 0.254 * mm, 0.0037, 5.8e7), verbose=False, method="hmatrix")
    op = HOperator(sol, verbose=False, leaf=32)
    g, zs = sol.greens(7e9), sol.surface_impedance(7e9)
    x = np.random.default_rng(0).standard_normal(m.n_unknowns) * (1 + 0.5j)
    y0, y1 = both(lambda: op.assemble(g, zs).matvec(x))
    assert op.stats["far_entries"] > 0
    assert np.linalg.norm(y1 - y0) < 1e-12 * np.linalg.norm(y0)
