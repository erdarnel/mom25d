import numpy as np
import pytest

from mom25d import Layout, Solver, Substrate, mm
from mom25d.hmatrix import HOperator, _partition, _tree
from mom25d.lumped import C

ER, H, W = 3.66, 0.254 * mm, 0.5561 * mm


def test_partition_covers_every_entry_once():
    """Near leaf pairs + far blocks (each unordered pair once, applied with
    its transpose) tile the matrix exactly -- also on a regular grid, where
    equal cluster diameters made the partition asymmetric before."""
    for pos in (np.random.default_rng(1).random((400, 3)),
                np.c_[np.mgrid[0:20, 0:20].reshape(2, -1).T * 1.0, np.zeros(400)]):
        n = len(pos)
        nodes = []
        root = _tree(np.arange(n), pos, np.full(n, 0.01), 16, nodes)
        far, near = [], []
        _partition(root, root, 1.0, far, near)
        cover = np.zeros((n, n), int)
        for a, b in near:
            cover[np.ix_(a.idx, b.idx)] += 1
        seen = set()
        for a, b in far:
            key = (min(a.id, b.id), max(a.id, b.id))
            if key in seen:
                continue
            seen.add(key)
            cover[np.ix_(a.idx, b.idx)] += 1
            cover[np.ix_(b.idx, a.idx)] += 1
        assert far and np.all(cover == 1)


def gcpw(length=6 * mm, lumped=False):
    lay = Layout()
    if lumped:
        lay.add_rect(-length / 2, -W / 2, -0.25 * mm, W / 2)
        lay.add_rect(0.25 * mm, -W / 2, length / 2, W / 2)
        lay.add_lumped("C1", (-0.25 * mm, 0), (0.25 * mm, 0), W, C(1e-12))
    else:
        lay.add_rect(-length / 2, -W / 2, length / 2, W / 2)
    lay.add_port("P1", (-length / 2, 0), "-x", W, 6 * mm)
    lay.add_port("P2", (length / 2, 0), "+x", W, 6 * mm)
    E, y1 = length / 2 + 6 * mm, W / 2 + 0.2 * mm + 1.0 * mm
    lay.add_pour((-E, -y1, E, y1), 0.2 * mm, via_pitch=1 * mm, via_drill=0.3 * mm, via_pad=0.5 * mm)
    return lay.mesh(0.3 * mm, 0.06 * mm, kind="tri")


def test_matvec_matches_dense():
    m = gcpw()
    sub = Substrate(ER, H, 0.0037, 5.8e7)
    sol = Solver(m, sub, verbose=False, method="hmatrix")
    op = HOperator(sol, verbose=False, leaf=32)
    g = sol.greens(5e9)
    op.assemble(g, sol.surface_impedance(5e9))
    Zd = Solver(m, sub, verbose=False, method="dense").impedance(5e9)
    x = np.random.default_rng(0).standard_normal(m.n_unknowns) + 1j * np.random.default_rng(1).standard_normal(m.n_unknowns)
    assert np.linalg.norm(op.matvec(x) - Zd @ x) < 1e-4 * np.linalg.norm(Zd @ x)
    assert op.stats["far_entries"] > 0


@pytest.mark.parametrize("lumped", [False, True])
def test_s_matches_dense(lumped):
    """GCPW line with a via fence (and a series 1 pF): S from the compressed
    operator + GMRES equals the dense solve."""
    m = gcpw(lumped=lumped)
    sub = Substrate(ER, H, 0.0037, 5.8e7)
    f = np.array([3e9, 9e9])
    Sd = Solver(m, sub, verbose=False, method="dense").sweep(f, interp=None).S
    Sh = Solver(m, sub, verbose=False, method="hmatrix").sweep(f).S
    assert np.abs(Sd - Sh).max() < 2e-4


def test_auto_method():
    m = gcpw()
    assert Solver(m, Substrate(ER, H), verbose=False).method == "dense"
    old = Solver.HMATRIX_N
    try:
        Solver.HMATRIX_N = 100
        assert Solver(m, Substrate(ER, H), verbose=False).method == "hmatrix"
    finally:
        Solver.HMATRIX_N = old
