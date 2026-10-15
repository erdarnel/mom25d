"""Synthetic board: one GCPW line, a drawn gap port at the input, a 0 ohm part (two pad ports 0.8 mm apart), a second
line, a GCPW feed at the output. Exercises config -> DXF -> sections -> layouts (mesh only, no solve) -> circuit ->
de-embedding algebra.   pytest -q Engine/tests/test_rfpcb.py"""
import json
import math

import ezdxf
import numpy as np
import pytest

from mom25d import rfpcb
from mom25d.rfpcb import circuit as CI
from mom25d.rfpcb import deembed as DE
from mom25d.rfpcb.engine import available
from mom25d.rfpcb.geometry import Board

W, G = 0.44, 0.40


def _rect(msp, layer, x0, y0, x1, y1):
    msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True, dxfattribs={"layer": layer})


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    d = tmp_path_factory.mktemp("board")
    doc = ezdxf.new()
    msp = doc.modelspace()
    _rect(msp, "L1_METAL", 0, -4, 20, 4)                                      # pour
    _rect(msp, "L1_NOT_METAL", 0.85, -W / 2 - G, 20, W / 2 + G)              # the line's gap area
    _rect(msp, "L1_METAL", 1.0, -W / 2, 19.0, W / 2)                          # the line (island in the gap area)
    msp.add_line((1.0, 0), (0.85, 0), dxfattribs={"layer": "GAP_PORTS"})
    msp.add_text("IN", dxfattribs={"layer": "GAP_PORTS", "insert": (0.9, 0.3)})
    for x, lab in ((9.0, "R1.1"), (9.8, "R1.2"), (19.0, "OUT")):
        msp.add_line((x, -W / 2), (x, W / 2), dxfattribs={"layer": "PORTS"})
        msp.add_text(lab, dxfattribs={"layer": "PORTS", "insert": (x + 0.02, W / 2 + 0.1)})
    for x in np.arange(0.4, 19.7, 0.4):
        for y in (-1.0, 1.0):
            msp.add_circle((x, y), 0.076, dxfattribs={"layer": "VIA_DRILL_L1-L3_D152"})
    for y in (-0.4, 0.0, 0.4):
        msp.add_circle((0.45, y), 0.076, dxfattribs={"layer": "VIA_DRILL_L1-L3_D152"})
    doc.saveas(d / "board.dxf")
    stack = {"layers": [
        {"kind": "mask", "name": "mask", "t_um": 20, "eps": 3.5, "tand": 0.02},
        {"kind": "cu", "name": "L1", "t_um": 35},
        {"kind": "diel", "name": "d12", "t_um": 100, "eps": 3.0, "tand": 0.003},
        {"kind": "cu", "name": "L2", "t_um": 18},
        {"kind": "diel", "name": "d23", "t_um": 100, "eps": 3.0, "tand": 0.003},
        {"kind": "cu", "name": "L3", "t_um": 18}],
        "vias": [{"from": "L1", "to": "L3", "drill_mm": 0.152, "pad_mm": 0.305, "dxf_layers": ["VIA_DRILL_L1-L3_D152"]}]}
    (d / "stackup.json").write_text(json.dumps(stack))
    f = np.arange(18e9, 42e9 + 1, 0.5e9)
    S = np.zeros((len(f), 2, 2), complex)
    S[:, 1, 0] = S[:, 0, 1] = np.exp(-1j * 2 * math.pi * f * 0.8e-3 * math.sqrt(2.3) / CI.C0)
    CI.write_s2p(d / "jumper.s2p", f, S)
    cfg = {"name": "synthetic", "dxf": "board.dxf", "stackup": "stackup.json", "out": "out",
           "chain": [{"line": "TL1", "from": "IN", "to": "R1.1"}, {"part": "R1", "s2p": "jumper.s2p"},
                     {"line": "TL2", "from": "R1.2", "to": "OUT"}],
           "ports": {"OUT": {"kind": "feed"}}, "model": {"ground_band": 2.0, "complete_band": 2.0}}
    (d / "board.json").write_text(json.dumps(cfg))
    return Board(rfpcb.load(d / "board.json"))


def test_ports(board):
    g = board.gaps["IN"]
    assert np.allclose(g["u"], (-1, 0)) and abs(g["w"] - W) < 1e-6 and abs(g["slot"] - 0.15) < 1e-9
    assert set(board.plines) == {"R1.1", "R1.2", "OUT"}
    assert [board.kind(x) for x in ("IN", "R1.1", "OUT")] == ["gap", "pad", "feed"]


def test_sections(board):
    s1, s2 = board.sections()
    assert (s1["name"], s2["name"]) == ("TL1", "TL2")
    assert np.allclose(s1["ports"][0][2], (-1, 0)) and np.allclose(s1["ports"][1][2], (1, 0))
    assert np.allclose(s2["ports"][0][2], (-1, 0)) and np.allclose(s2["ports"][1][2], (1, 0))
    assert abs(s1["rf"].area - W * 8.0) < 0.02 and abs(s2["rf"].area - W * 9.2) < 0.02
    for s in (s1, s2):
        assert s["gnd"].area > 10 and len(s["vias"]) > 10
        assert all(abs(p[4] - G) < 0.006 for p in s["ports"])
    assert "P1" in s1["gapports"] and s1["gapports"]["P1"]["land"].area > 0


def test_segments(board):
    seg = board.segments()
    assert abs(sum(x[0] for x in seg["TL1"]) - 8.0) < 0.02
    assert abs(sum(x[0] for x in seg["TL2"]) - 9.2) < 0.02
    assert all(abs(w - W) < 0.006 and abs(g - G) < 0.006 for L, w, g in seg["TL1"])


@pytest.mark.skipif(not available(), reason="MOM25D_PATH not set")
def test_layouts(board):
    from mom25d.rfpcb.place import Placer
    pl = Placer(board)
    for s in board.sections():
        lay, mesh = pl.build(s)
        assert mesh.n_unknowns > 200
    c = board.complete()
    lay, mesh = pl.build(c, band=2.0)
    names = sorted(q.name for q in lay.lumped)
    assert names == ["P1", "P2", "P3"] and [p.name for p in lay.ports] == ["P4"]
    st = board.stack.mom()
    assert st is not None


def test_qs2d_and_circuit(board):
    r = CI.qs2d(W, G, **board.stack.qs2d_params())
    assert 30 < r["Z0"] < 80 and 1.5 < r["eeff"] < 3.0
    S = CI.Circuit(board).chain()
    assert np.all(np.isfinite(S)) and np.all(np.abs(S[1, 0]) < 1.0)


def _rand_recip(rng):
    A, B, C = rng.normal(1, .1) + 1j * rng.normal(0, .1), 50 * (rng.normal(0, .1) + 1j * rng.normal(0, .3)), \
        (rng.normal(0, .1) + 1j * rng.normal(0, .3)) / 50
    return np.array([[A, B], [C, (1 + B * C) / A]])


def test_deembed_algebra():
    rng = np.random.default_rng(1)
    X1, X2, D = _rand_recip(rng), _rand_recip(rng), _rand_recip(rng)
    M = X1 @ D @ DE.rev(X2)
    Sd = DE.deembed_nport(DE.a2s(M), [X1, X2])
    assert np.abs(Sd - DE.a2s(D)).max() < 1e-10
    # terminate == ABCD cascade: 4-port of two independent 2-ports, a part between ports 2 and 3
    A1, A2, P = _rand_recip(rng), _rand_recip(rng), _rand_recip(rng)
    S4 = np.zeros((4, 4), complex)
    S4[:2, :2], S4[2:, 2:] = DE.a2s(A1), DE.a2s(A2)
    S2 = DE.terminate(S4, (1, 2), DE.a2s(P))
    assert np.abs(S2 - DE.a2s(A1 @ P @ A2)).max() < 1e-10
    # fit_box recovers a box from two line lengths
    gam, z0 = 5 + 900j, 48.0
    X = _rand_recip(rng)
    Ms = [X @ DE.tline(z0, gam, L) @ DE.rev(X) for L in (1.5, 3.0)]
    Xf, r = DE.fit_box(Ms, [DE.tline(z0, gam, L) for L in (1.5, 3.0)])
    pred = DE.a2s(Xf @ DE.tline(z0, gam, 4.5) @ DE.rev(Xf))
    assert r < 1e-8 and np.abs(pred - DE.a2s(X @ DE.tline(z0, gam, 4.5) @ DE.rev(X))).max() < 1e-6


@pytest.mark.skipif(not available(), reason="mom25d not importable")
def test_plane_references(board):
    """With an inner plane: reference-line voids per port type, feed refs end on a gap port (Z0 by reciprocity), the
    plane is voided under feeds, and section builds go back to the DXF voids."""
    from shapely.geometry import box
    from mom25d.rfpcb.models import Project
    pr = Project(board.cfg)
    pr.pl.plane = "L2"
    pr.pl._void = box(0, 0, 0, 0)          # no DXF L2 in the synthetic board: an empty void
    pv = pr.ref_void(("pad", W, G, 0.0), 3.0)
    gv = pr.ref_void(("pour", W, G, 0.3), 3.0)
    assert pv.bounds[2] > 3.0 / 2 + 1.0 and abs(gv.bounds[2] - (1.5 + 0.30)) < 1e-9
    assert abs(pv.bounds[3] - (W / 2 + G + 0.05)) < 1e-9
    lay = pr.feed_ref(("pad", W, G, 0.0))
    assert [p.name for p in lay.ports] == ["P1"] and [q.name for q in lay.lumped] == ["P2"]
    assert lay.apertures and lay.apertures[0][2].bounds[0] < -(1.5 + 3.0) * 1e-3   # opening under the feed
    jobs = pr.jobs()
    sec_job = next(j for j in jobs if j[0].startswith("sec_"))
    sec_job[1]()
    assert pr.pl.void_override is None
