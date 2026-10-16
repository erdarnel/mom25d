"""Oscillation risk screen (mom25d.oscillation): loop detection and closed-form couplings on explicit paths."""
import json
import math

import numpy as np
import pytest

from mom25d import oscillation as osc


def _cfg(tmp_path, path, layers=None, cavities=None, parts=None, corners=None, **kw):
    d = dict(name="test", out=str(tmp_path / "out"),
             layers=layers or {"L1": {"h": 0.2, "eeff": 2.5}, "L4": {"h": 0.2, "eeff": 2.5}},
             cavities=cavities or {},
             parts=parts or {"amp": {"gain_db": 30.0, "active": True, "tc_db_per_c": -0.02},
                             "pad": {"gain_db": -3.0}},
             corners=corners or [{"f": 10.0, "t": 25.0}],
             channels=[{"name": "ch1", "path": path}], **kw)
    p = tmp_path / "osc.json"
    p.write_text(json.dumps(d))
    return osc.load(p)


def _hairpin(gap, layer2="L1", gain="amp"):
    return [{"line": "L1", "points": [[0, 0], [20, 0]]}, {"part": gain, "label": "A1"},
            {"line": layer2, "points": [[20, gap], [0, gap]]}]


def _run(cfg):
    chains, _ = osc.trace(cfg)
    return osc.assess_all(chains, cfg)


def test_line_line_loop_matches_formula(tmp_path):
    cfg = _cfg(tmp_path, _hairpin(2.0))
    rows = _run(cfg)[0]["rows"]
    ll = [r for r in rows if r["kind"] == "line-line"]
    assert ll, "no line-line loop on a parallel hairpin with gain"
    kL = osc.coupling_kL(2.0, 0.2, 50.0, 2.5)
    beta = osc.beta_mm(10e9, 2.5)
    kb = (1 + 0.7) / 4 * kL
    kf = (1 - 0.7) / 2 * beta * kL * 20.0
    expect = 30.0 + 20 * math.log10(max(kb, kf))
    assert abs(ll[0]["loop"] - expect) < 0.5
    assert ll[0]["agg_stage"] == "after A1" and ll[0]["vic_stage"] == "before A1"


def test_loop_falls_with_spacing_and_needs_gain_and_same_layer(tmp_path):
    near = _run(_cfg(tmp_path, _hairpin(1.0)))[0]["rows"][0]["loop"]
    far = _run(_cfg(tmp_path, _hairpin(5.0)))[0]["rows"][0]["loop"]
    assert near > far + 6
    assert not _run(_cfg(tmp_path, _hairpin(2.0, gain="pad")))[0]["rows"]          # no gain stage: no loop
    assert not _run(_cfg(tmp_path, _hairpin(2.0, layer2="L4")))[0]["rows"]         # other (shielded) layer


def test_corners_temperature_and_states(tmp_path):
    parts = {"amp": {"gain_db": 30.0, "active": True, "tc_db_per_c": -0.02},
             "dsa": {"states": {"0dB": -1.0, "10dB": -11.0}, "state": "0dB"}}
    path = [{"line": "L1", "points": [[0, 0], [20, 0]]}, {"part": "amp"}, {"part": "dsa"},
            {"line": "L1", "points": [[20, 2], [0, 2]]}]
    cfg = _cfg(tmp_path, path, parts=parts,
               corners=[{"f": 10.0, "t": 25.0}, {"f": 10.0, "t": -40.0}, {"f": 10.0, "t": 25.0, "states": {"dsa": "10dB"}}])
    w = [R["rows"][0]["loop"] for R in _run(cfg)]
    assert abs((w[1] - w[0]) - 1.3) < 0.05          # -0.02 dB/degC x -65 degC
    assert abs((w[0] - w[2]) - 10.0) < 0.05


def test_channel_and_corner_expansion():
    ch = osc.expand_channels([{"repeat": {"n": [1, 2]}, "name": "ch{n}", "path": [{"line": "L1", "net": "RF_{n}"}]}])
    assert [c["name"] for c in ch] == ["ch1", "ch2"] and ch[1]["path"][0]["net"] == "RF_2"
    co = osc.expand_corners({"f": [10, 20], "t": [25, -40]})
    assert len(co) == 4 and co[0]["label"].startswith("10 GHz")


def test_via_cavity_coupling(tmp_path):
    path = [{"line": "L1", "points": [[0, 0], [1, 0]]}, {"via": {}}, {"line": "L4", "points": [[1, 0], [2, 0]]},
            {"part": "amp"}, {"line": "L4", "points": [[2, 0], [10, 0]]}, {"via": {}},
            {"line": "L1", "points": [[10, 0], [11, 0]]}]
    cav = {"P": {"eps": 4.0, "d": 0.2}}
    cfg = _cfg(tmp_path, path, cavities=cav)
    rows = [r for r in _run(cfg)[0]["rows"] if r["kind"] == "via cavity"]
    assert len(rows) == 1
    c = 20 * math.log10(osc.cavity_coupling(9.0, 10e9, 4.0, 0.2))
    assert abs(rows[0]["c"] - c) < 1e-6
    assert abs(rows[0]["dg"] - (30.0 - 8 * 0.0)) < 0.2


def test_fence_transmission():
    xy = np.array([0.0, 0.0])
    assert osc.fence_transmission(xy, np.zeros((0, 3)), 10.0)[0] == 1.0
    a = np.linspace(0, 2 * np.pi, 25)[:-1]
    ring = np.c_[1.0 * np.cos(a), 1.0 * np.sin(a), np.full(len(a), 0.15)]      # 0.26 mm pitch
    T, n, gmax = osc.fence_transmission(xy, ring, 10.0)
    assert n == 24 and T < 0.1
    sparse = ring[::6]                                                       # 4 vias, 1.4 mm gaps
    assert osc.fence_transmission(xy, sparse, 10.0)[0] > T


def test_outputs(tmp_path):
    cfg = _cfg(tmp_path, _hairpin(2.0), corners={"f": [5.0, 10.0], "t": 25})
    res = _run(cfg)
    md = osc.write_markdown(res, cfg, tmp_path / "o.md")
    html = osc.write_html(res, cfg, {}, tmp_path / "o.html")
    js = osc.write_json(res, tmp_path / "o.json")
    assert "Oscillation screen" in open(md).read()
    h = open(html).read()
    assert "__DATA__" not in h and '"chains":["ch1"]' in h
    assert json.load(open(js))[0]["summary"]["line-line"]["n"] >= 1


def test_errors(tmp_path):
    with pytest.raises(ValueError):
        osc.trace(_cfg(tmp_path, [{"via": {}}]))
    with pytest.raises(ValueError):
        osc.trace(_cfg(tmp_path, [{"part": "nope"}]))


def test_parallel_branches(tmp_path):
    """A split / combine section (balanced amplifier pair): shared lines kept once, levels split and recombined,
    loops found into the second branch."""
    br = lambda y: [{"line": "L1", "points": [[0, 0], [5, 0], [5, y], [15, y]]}, {"part": "amp", "label": "A1"},  # noqa: E731
                    {"line": "L1", "points": [[15, y], [20, y], [20, 0], [30, 0]]}]
    path = [{"parallel": [br(3), br(-3)], "label": "pair"},
            {"line": "L1", "points": [[30, 0], [30, -5], [6, -5]]}]
    cfg = _cfg(tmp_path, path)
    chains, _ = osc.trace(cfg)
    kinds = [it[0] for it in chains["ch1"]]
    assert kinds == ["line", "parallel", "line", "line"]
    assert np.allclose(chains["ch1"][0][2], [[0, 0], [5, 0]]) and np.allclose(chains["ch1"][2][2], [[20, 0], [30, 0]])
    res = osc.assess_all(chains, cfg)
    S = res[0]["samples"]["ch1"]
    assert set(S["branch"]) == {0, 1, 2}
    assert abs(S["end_level"] - 30.0) < 1e-6                     # -3 dB split, +30 dB, ideal in-phase combine
    assert abs(S["level"][S["branch"] == 2].min() + 10 * np.log10(2)) < 1e-6
    ll = [r for r in res[0]["rows"] if r["kind"] == "line-line"]
    into = [r for r in ll if S["branch"][r["vic"][0]] == 2 and S["branch"][r["agg"][0]] == 0]
    assert into, "no loop into the lower branch, 2 mm from the return"
    # victim in one branch of two: path gain = 30 dB amp - 3 dB combine of one input (level difference says +33 dB)
    assert abs(max(r["dg"] for r in into) - (30.0 - 10 * np.log10(2))) < 0.05
