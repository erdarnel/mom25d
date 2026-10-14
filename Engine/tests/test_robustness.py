"""Sweep checkpoint / resume, memory pre-flight, mesh morphing (Layout.mesh(like=...)), viewer theme."""
import os

import numpy as np
import pytest

from mom25d import GHz, Layout, Solver, Substrate, mm
from mom25d import freqinterp, view3d

pytest.importorskip("triangle")
pytest.importorskip("shapely")
from shapely.geometry import Polygon  # noqa: E402

W, ER, H = 0.6 * mm, 3.66, 0.254 * mm


def notch(ls=6 * mm, dx=0.0):
    """Line with an open stub drawn as a polygon (length ls, shifted dx along x)."""
    lay = Layout()
    lay.add_rect(-W / 2 - 1 * mm, -W / 2, W / 2 + 1 * mm, W / 2)
    lay.add_polygon(Polygon([(-W / 2 + dx, W / 2), (W / 2 + dx, W / 2), (W / 2 + dx, W / 2 + ls),
                             (-W / 2 + dx, W / 2 + ls)]))
    lay.add_port("P1", (-W / 2 - 1 * mm, 0), "-x", W, 6 * mm)
    lay.add_port("P2", (W / 2 + 1 * mm, 0), "+x", W, 6 * mm)
    return lay


def test_checkpoint_resume(tmp_path):
    f = np.array([3.0, 5.0, 7.0]) * GHz
    m = notch().mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15)
    sol = Solver(m, Substrate(ER, H), verbose=False)
    ck = str(tmp_path / "ck.npz")
    full = sol.sweep(f, interp=None, checkpoint=ck)
    d = dict(np.load(ck))
    assert d["done"].all()
    # pretend the run was killed after the first two frequencies
    d["done"][2] = False
    d["S"][2] = 0
    np.savez(ck, **d)
    calls = []
    orig = sol.solve_currents
    sol.solve_currents = lambda *a, **k: calls.append(a[0]) or orig(*a, **k)
    again = sol.sweep(f, interp=None, checkpoint=ck)
    assert calls == [f[2]]                       # only the missing frequency is solved
    np.testing.assert_allclose(again.S, full.S, atol=1e-12)
    # a different sweep does not resume from it
    calls.clear()
    sol.sweep(f[:2], interp=None, checkpoint=ck)
    assert len(calls) == 2


def test_memory_preflight(monkeypatch):
    m = notch().mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15)
    sol = Solver(m, Substrate(ER, H), verbose=False)
    need = sol.memory_estimate()
    assert need == 3 * 16 * m.n_unknowns ** 2
    monkeypatch.setattr(freqinterp, "_available_memory", lambda: need / 10)
    with pytest.raises(MemoryError):
        sol.sweep([3 * GHz], interp=None, wait_memory=0.05)
    r = sol.sweep([3 * GHz], interp=None, wait_memory=0)      # warn only
    assert np.isfinite(r.S).all()


def test_morph_keeps_topology_and_tracks_geometry():
    lay0, lay1 = notch(6 * mm), notch(6 * mm + 20e-6, dx=10e-6)
    m0 = lay0.mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15)
    m1 = lay1.mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15, like=m0)
    assert m1.morphed and np.array_equal(m1.tris, m0.tris) and m1.n_unknowns == m0.n_unknowns
    assert m1.area.sum() == pytest.approx(m0.area.sum() + W * 20e-6, rel=1e-9)   # exact copper area
    # incompatible change (extra polygon vertex): fresh mesh
    lay2 = notch(6 * mm)
    lay2.polys[0] = Polygon([(-W / 2, W / 2), (0, W / 2), (W / 2, W / 2), (W / 2, W / 2 + 6 * mm), (-W / 2, W / 2 + 6 * mm)])
    assert not lay2.mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15, like=m0).morphed


def test_morph_smooth_response():
    """A 20 um stub change moves the notch smoothly on the morphed mesh, and agrees with a fresh mesh."""
    f = np.array([5.5, 6.0, 6.5]) * GHz
    sub = Substrate(ER, H)
    m0 = notch(8 * mm).mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15)
    s = []
    for dl in (0.0, 20e-6, 40e-6):
        m = notch(8 * mm + dl).mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15, like=m0)
        assert m.morphed
        s.append(Solver(m, sub, verbose=False).sweep(f, interp=None).S[:, 1, 0])
    s = np.array(s)
    # equal steps -> equal changes (second difference << first difference)
    d1, d2 = s[1] - s[0], s[2] - 2 * s[1] + s[0]
    assert np.all(np.abs(d2) < 0.05 * np.abs(d1) + 1e-6)
    fresh = Solver(notch(8 * mm + 40e-6).mesh(0.6 * mm, 0.08 * mm, kind="tri", min_angle=15), sub,
                   verbose=False).sweep(f, interp=None).S[:, 1, 0]
    np.testing.assert_allclose(np.abs(s[2]), np.abs(fresh), atol=0.02)


def test_viewer_theme():
    tpl = '@media (prefers-color-scheme: dark) { x } window.matchMedia("(prefers-color-scheme: dark)").matches'
    assert view3d._theme(tpl, "auto") == tpl
    d = view3d._theme(tpl, "dark")
    assert "@media all" in d and d.endswith("true")
    li = view3d._theme(tpl, "light")
    assert "@media not all" in li and li.endswith("false")
    with pytest.raises(ValueError):
        view3d._theme(tpl, "blue")
