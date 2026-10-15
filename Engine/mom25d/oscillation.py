"""Oscillation risk screen for RF PCBs with gain stages.

Every RF path (one "chain" per channel) is traced along the copper centre line and given a signal level: line loss
per mm, part gains and losses, combiner splits. A feedback loop exists wherever a downstream piece of the same
signal (aggressor, level La) couples to an upstream piece (victim, level Lv) with at least one gain stage between
them. Loop gain = (La - Lv) + C, with C the coupling between them; oscillation needs loop gain >= 0 dB at a
frequency where the amplifiers have gain. A loop at -20 dB gives about +-0.9 dB ripple.

Coupling mechanisms (closed-form, each an upper bound; meant to rank risks and pick the few couplings that need an
EM model, not to replace it):

  line-line      two parallel microstrips on the same layer, centre spacing d over height h (quasi-static, no side
                 grounds): kL = ln(1 + (2h/d)^2) / (4 pi L / mu0), L = Z0 sqrt(eeff) / c, kC = kc_ratio kL;
                 backward (saturated) Kb = (kL + kC) / 4, forward Kf = (kL - kC) / 2 * beta * coupled length,
                 runs summed coherently along the aggressor (worst case). Solid planes between layers are assumed
                 to shield layers from each other.
  via radiation  the vertical current of a signal via above / below its reference plane radiates as a current
                 element with its ground image; its H field links the line-to-ground loop of upstream lines (x2 for
                 the electric part), summed along the victim with phase; times the transmission of the ground-via
                 fence around the via.
  via cavity     the via section inside a plane-pair cavity launches a radial parallel-plate wave that reaches
                 upstream signal vias: Z21 = (eta k d / 4) |H0(2)(k r)|, C = Z21 / 2 Z0, times both vias' fence
                 transmission (each angular gap between fence vias = a below-cut-off channel one drill long).

Not modelled: coupling between different chains (e.g. channels that meet in a combiner), loops through bias or
control lines, part-internal feedback, phase (every loop is taken at its worst phase).

Use (config file format: see `load` and Engine/docs/oscillation.md):

    python -m mom25d.oscillation board_osc.json            -> <out>/oscillation.{json,md,html}

or from Python: `cfg = load(path); chains = trace(cfg); res = assess_all(chains, cfg); write_outputs(...)`.
"""
import copy
import json
import math
import os
from pathlib import Path

import numpy as np

C0 = 299792458.0
MU0 = 4e-7 * math.pi

DEFAULTS = {
    "name": "board",
    "out": "oscillation",
    "step": 0.25,            # mm, path sampling
    "dmax": 15.0,            # mm, largest line-line spacing considered
    "dmin": 0.5,             # mm, smaller spacings are taken as the same piece of copper (e.g. at a part)
    "parallel_cos": 0.7,     # line-line: |cos| of the angle between the two runs at least this
    "run_gap": 8.0,          # mm, victim jump that still continues a coupled run
    "rvia": 40.0,            # mm, via radiation: victim lines within this distance
    "rvia_min": 0.8,         # mm, ... and beyond this one
    "fence_rmin": 0.35, "fence_rmax": 3.0,   # mm, ground vias counted as the fence round a signal via
    "via_loss_db": 0.15,
    "t_ref": 25.0,           # degC at which part gains are given
    "fence_nets": ["GND"],
    "signal_via_drill_layers": None,   # drill layers of the signal vias "via" endpoints look for (None: all)
    "display_nets": None,    # copper shown in the viewer besides the traced nets (None: fence_nets)
    "display_band": 3.0,     # mm round the paths
    "simplify": 0.01,        # mm, viewer copper simplification
    "top_rows": 40,          # loops listed per corner in the Markdown report
    "corners": [],
    "layers": {},
    "cavities": {},
    "parts": {},
    "channels": [],
}
LAYER_DEFAULTS = {"t": 0.035, "z0": 50.0, "loss_db_per_mm": 0.0, "loss_f_ghz": 10.0, "kc_ratio": 0.7,
                  "coupling_offset_db": 0.0, "fence_drill_layers": None, "eps": None, "via_height": None,
                  "side": None}


# --------------------------------------------------------------------------- config

class Config(dict):
    def __init__(self, d, root):
        super().__init__(copy.deepcopy(DEFAULTS))
        self.update(d)
        self.root = Path(root)
        for name, L in self["layers"].items():
            for k in ("h", "eeff"):
                if k not in L:
                    raise ValueError(f"layer {name}: '{k}' missing")
            full = dict(LAYER_DEFAULTS, **L)
            full["eps"] = full["eps"] or full["eeff"]
            full["via_height"] = full["via_height"] or full["h"] + 0.5 * full["t"]
            self["layers"][name] = full
        if self["display_nets"] is None:
            self["display_nets"] = list(self["fence_nets"])
        self["channels"] = expand_channels(self["channels"])
        self["corners"] = expand_corners(self["corners"])
        if not self["corners"]:
            raise ValueError("no corners: give e.g. \"corners\": {\"f\": [17.7, 21.2], \"t\": [25, -40]}")

    def path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def out(self):
        o = self.path(self["out"])
        o.mkdir(parents=True, exist_ok=True)
        return o


def _subst(x, env):
    if isinstance(x, str):
        for k, v in env.items():
            x = x.replace("{" + k + "}", str(v))
        return x
    if isinstance(x, list):
        return [_subst(v, env) for v in x]
    if isinstance(x, dict):
        return {k: _subst(v, env) for k, v in x.items()}
    return x


def expand_channels(ch):
    """[{"name", "path"}] from a list of channels and / or templates {"repeat": {"n": [1, 2]}, "name": "ch{n}",
    "path": [...]} ("{n}" replaced in every string)."""
    out = []
    for c in (ch if isinstance(ch, list) else [ch]):
        if "repeat" in c:
            keys = list(c["repeat"])
            vals = list(zip(*[c["repeat"][k] for k in keys]))
            for v in vals:
                env = dict(zip(keys, v))
                out.append(_subst({k: c[k] for k in c if k != "repeat"}, env))
        else:
            out.append(c)
    for k, c in enumerate(out):
        c.setdefault("name", f"ch{k + 1}")
    return out


def expand_corners(c):
    """[{"f": GHz, "t": degC, "states": {...}, "label"}] from a list, or {"f": [...], "t": [...], "states": {...}}
    (every f with every t)."""
    if isinstance(c, dict):
        fs = c["f"] if isinstance(c["f"], list) else [c["f"]]
        ts = c.get("t", 25.0)
        ts = ts if isinstance(ts, list) else [ts]
        c = [dict(f=f, t=t, states=c.get("states", {})) for t in ts for f in fs]
    out = []
    for x in c:
        x = dict(x)
        x.setdefault("t", 25.0)
        x.setdefault("states", {})
        x.setdefault("label", f"{x['f']:g} GHz, {x['t']:g} °C" + "".join(f", {k} {v}" for k, v in x["states"].items()))
        out.append(x)
    return out


def load(path):
    """Read a JSON config. Lengths in mm, frequencies in GHz, temperatures in degC; paths relative to the file.

    {
      "name": "Rx board", "out": "osc", "odb": "odb_tree/odb",          (an extracted ODB++ tree; optional if
                                                                        every line gives "points")
      "layers": {"top_layer": {"h": 0.185, "t": 0.06, "eeff": 2.64, "eps": 3.54, "z0": 50,
                               "loss_db_per_mm": 0.024, "loss_f_ghz": 19.45,
                               "fence_drill_layers": ["drill_top-bottom", "drill_top-l2"]}},
      "cavities": {"L10-L11": {"eps": 4.21, "d": 0.204, "fence_drill_layers": ["drill_top-bottom"]}},
      "parts": {"lna": {"gain_db": 28.5, "active": true, "tc_db_per_c": -0.024},    (or "s2p": file, |S21|;
               "sw": {"gain_db": -1.6}, "dsa": {"states": {"0dB": -2.0, "15dB": -17}, "state": "0dB"}},
                                                                        or "gain_vs_f": [[GHz, dB], ...])
      "corners": {"f": [17.7, 21.2], "t": [25, -40]},
      "channels": [{"repeat": {"n": [1, 2]}, "name": "ch{n}", "path": [
          {"part": "conn"},
          {"line": "top_layer", "net": "RF_IN_{n}", "from": "J1_{n}", "to": "U1*"},
          {"part": "lna", "label": "LNA1"},
          {"line": "top_layer", "net": "LNA_OUT_{n}", "from": "U1*", "to": "via"},
          {"via": {}},
          {"line": "bottom_layer", "net": "LNA_OUT_{n}", "from": "via", "to": "J2*",
           "split_at": [{"part": "wilk2", "near": "R89"}]},
          {"line": "bottom_layer", "points": [[x, y], ...]}                  (explicit centre line)
      ]}]
    }

    Line endpoints: "REF" (a pin of that refdes on the line's net), "REF.PIN", "PART*" (a pin on the net of a part
    whose part number starts with PART), "via" / "via:NET" (a signal via of the net), or [x, y]. "from" prefers the
    point where the previous line ended (same refdes or via); "to" takes the candidate other than "from" with the
    shortest centre-line path. A tc_db_per_c is the gain change per degC (negative for a gain that rises cold).
    """
    path = Path(path)
    return Config(json.loads(path.read_text()), path.parent)


# --------------------------------------------------------------------------- geometry front-end

class CentreLine:
    """Medial axis of one connected copper polygon (Voronoi of the densified outline, edges inside the copper)."""

    def __init__(self, geom, step=0.02):
        import shapely
        from scipy.sparse import coo_matrix
        from scipy.spatial import Voronoi

        self.geom = geom
        pts = shapely.get_coordinates(shapely.segmentize(geom.boundary, step))
        vor = Voronoi(pts)
        V = vor.vertices
        inside = shapely.contains_xy(geom, V[:, 0], V[:, 1])
        rv = np.array([r for r in vor.ridge_vertices if -1 not in r])
        rv = rv[inside[rv[:, 0]] & inside[rv[:, 1]]]
        mid = 0.5 * (V[rv[:, 0]] + V[rv[:, 1]])
        rv = rv[shapely.contains_xy(geom, mid[:, 0], mid[:, 1])]
        w = np.linalg.norm(V[rv[:, 0]] - V[rv[:, 1]], axis=1)
        n = len(V)
        self.G = coo_matrix((np.r_[w, w], (np.r_[rv[:, 0], rv[:, 1]], np.r_[rv[:, 1], rv[:, 0]])),
                            shape=(n, n)).tocsr()
        self.nodes, self.V = np.unique(rv), V

    def _node(self, a):
        P = self.V[self.nodes]
        return self.nodes[np.argmin(np.linalg.norm(P - a, axis=1))]

    def length(self, a, b):
        from scipy.sparse.csgraph import dijkstra

        ia, ib = self._node(a), self._node(b)
        d = dijkstra(self.G, indices=ia, min_only=True)[ib] if ia != ib else 0.0
        return d + np.linalg.norm(self.V[ia] - a) + np.linalg.norm(self.V[ib] - b)

    def path(self, a, b):
        """Centre-line polyline (n, 2) from a to b."""
        from scipy.sparse.csgraph import dijkstra

        ia, ib = self._node(a), self._node(b)
        _, pred = dijkstra(self.G, indices=ia, return_predecessors=True)
        seq = [ib]
        while seq[-1] != ia:
            if pred[seq[-1]] < 0:
                raise ValueError(f"no centre-line path from {tuple(np.round(a, 3))} to {tuple(np.round(b, 3))}")
            seq.append(pred[seq[-1]])
        return np.vstack([a, self.V[seq[::-1]], b])


class OdbBoard:
    """Copper, pins and vias of an ODB++ export for tracing chains."""

    def __init__(self, cfg):
        from .odb import ODB, components

        self.cfg = cfg
        cache = cfg.path(cfg["odb_cache"]) if cfg.get("odb_cache") else cfg.out / "cache"
        self.odb = ODB(str(cfg.path(cfg["odb"])), cache_dir=str(cache))
        self.comps = components(self.odb, "top") + components(self.odb, "bottom")
        self.by_ref = {c["refdes"]: c for c in self.comps}
        self._net, self._vias = {}, None
        self._lines = {}

    def vias(self, net=None):
        if self._vias is None:
            self._vias = self.odb.all_vias()
        dl = self.cfg["signal_via_drill_layers"]
        return [v for v in self._vias if (net is None or v["net"] == net) and (dl is None or v["drill_layer"] in dl)]

    def fence(self, drill_layers):
        if self._vias is None:
            self._vias = self.odb.all_vias()
        fn = set(self.cfg["fence_nets"])
        return np.array([(v["x"], v["y"], v["drill"]) for v in self._vias if v["net"] in fn and
                         (drill_layers is None or v["drill_layer"] in drill_layers)], float).reshape(-1, 3)

    def net_copper(self, layer, net):
        key = (layer, net)
        if key not in self._net:
            import shapely

            feats = self.odb.layer_features(layer)
            idx = self.odb.net_features(layer).get(net)
            if not idx:
                raise ValueError(f"net {net} has no copper on {layer}")
            self._net[key] = shapely.union_all([feats[i] for i in idx if i < len(feats) and feats[i] is not None])
        return self._net[key]

    def centreline(self, layer, net, near):
        import shapely
        from shapely.geometry import Point

        g = self.net_copper(layer, net)
        piece = min(shapely.get_parts(g), key=lambda p: p.distance(Point(near)))
        key = piece.wkb
        if key not in self._lines:
            self._lines[key] = CentreLine(piece)
        return self._lines[key]

    def candidates(self, spec, net):
        """[(xy, tag)] for an endpoint spec on `net` (tag: refdes, 'via@x,y' or None)."""
        if isinstance(spec, (list, tuple)):
            return [(np.asarray(spec, float), None)]
        if spec.startswith("via"):
            vnet = spec.split(":", 1)[1] if ":" in spec else net
            return [(np.array([v["x"], v["y"]]), f"via@{v['x']:.3f},{v['y']:.3f}") for v in self.vias(vnet)]
        pin = None
        if "." in spec and spec.split(".")[0] in self.by_ref:
            spec, pin = spec.split(".", 1)
        if spec in self.by_ref:
            comps = [self.by_ref[spec]]
        else:
            pre = spec.rstrip("*")
            comps = [c for c in self.comps if c["part"].startswith(pre)]
        return [(np.array([p["x"], p["y"]]), c["refdes"]) for c in comps for p in c["pins"]
                if p["net"] == net and (pin is None or p["name"] == pin)]


def _same(a, b, tol=0.05):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) < tol


def trace(cfg, board=None):
    """Chains as lists of items: ("line", layer, P (n, 2), net), ("part", key, label, xy), ("via", xy, opts)."""
    need = any("line" in e and "points" not in e for c in cfg["channels"] for e in c["path"])
    if board is None and need:
        board = OdbBoard(cfg)
    chains = {}
    for c in cfg["channels"]:
        items, prev_xy, prev_tag = [], None, None
        for e in c["path"]:
            if "part" in e:
                if e["part"] not in cfg["parts"]:
                    raise ValueError(f"{c['name']}: part '{e['part']}' not in parts")
                items.append(("part", e["part"], e.get("label", e["part"]), prev_xy))
            elif "via" in e:
                if prev_xy is None:
                    raise ValueError(f"{c['name']}: a via must follow a line")
                items.append(("via", prev_xy, dict(e["via"] or {})))
            elif "line" in e:
                layer = e["line"]
                if layer not in cfg["layers"]:
                    raise ValueError(f"{c['name']}: layer {layer} not in layers")
                if "points" in e:
                    P = np.asarray(e["points"], float)
                    a_tag = b_tag = None
                else:
                    net = e["net"]
                    ca = board.candidates(e["from"], net)
                    if not ca:
                        raise ValueError(f"{c['name']}: no '{e['from']}' on net {net}")
                    if len(ca) > 1 and prev_xy is not None:
                        hit = [x for x in ca if _same(x[0], prev_xy) or (prev_tag and x[1] == prev_tag)]
                        ca = hit or sorted(ca, key=lambda x: np.linalg.norm(x[0] - prev_xy))
                    elif len(ca) > 1:
                        raise ValueError(f"{c['name']}: '{e['from']}' on net {net} is ambiguous ({len(ca)} pins)")
                    a, a_tag = ca[0]
                    cb = [x for x in board.candidates(e["to"], net) if not _same(x[0], a)
                          and (x[1] is None or x[1] != a_tag or x[1].startswith("via"))]
                    if not cb:
                        raise ValueError(f"{c['name']}: no '{e['to']}' on net {net} other than the start")
                    cl = board.centreline(layer, net, a)
                    if len(cb) > 1:
                        cb = sorted(cb, key=lambda x: cl.length(a, x[0]))
                    b, b_tag = cb[0]
                    P = cl.path(a, b)
                splits = []
                for s in e.get("split_at", []):
                    near = s["near"]
                    xy = (np.array([board.by_ref[near]["x"], board.by_ref[near]["y"]])
                          if isinstance(near, str) else np.asarray(near, float))
                    splits.append((int(np.argmin(np.linalg.norm(P - xy, axis=1))), s))
                k0 = 0
                for k, s in sorted(splits, key=lambda t: t[0]):
                    items.append(("line", layer, P[k0:k + 1], e.get("net")))
                    items.append(("part", s["part"], s.get("label", s["part"]), P[k]))
                    k0 = k
                items.append(("line", layer, P[k0:], e.get("net")))
                prev_xy, prev_tag = P[-1], b_tag
        chains[c["name"]] = items
        print(f"traced {c['name']}: {sum(len(it[2]) for it in items if it[0] == 'line')} centre-line points", flush=True)
    return chains, board


# --------------------------------------------------------------------------- physics

def part_gain(cfg, key, f_ghz, t, states=None):
    """Gain (dB) of part `key` at f, temperature t (state from `states` or the part's default)."""
    spec = cfg["parts"][key]
    if "states" in spec:
        st = (states or {}).get(key, spec.get("state"))
        g = spec["states"][st] if st is not None else max(spec["states"].values())
    elif "s2p" in spec:
        cache = spec.setdefault("_s21", {})
        if "f" not in cache:
            from .rfpcb.circuit import read_touchstone

            f, S, _ = read_touchstone(cfg.path(spec["s2p"]))
            cache["f"], cache["g"] = f / 1e9, 20 * np.log10(np.abs(S[1, 0]) + 1e-30)
        g = float(np.interp(f_ghz, cache["f"], cache["g"]))
    elif "gain_vs_f" in spec:
        t_ = np.asarray(spec["gain_vs_f"], float)
        g = float(np.interp(f_ghz, t_[:, 0], t_[:, 1]))
    else:
        g = float(spec.get("gain_db", 0.0))
    return g + spec.get("tc_db_per_c", 0.0) * (t - cfg["t_ref"])


def coupling_kL(d, h, z0, eeff):
    """Inductive coupling factor of two lines at centre spacing d over height h (same units)."""
    L = z0 * math.sqrt(eeff) / C0
    return 1e-7 * np.log1p((2 * h / np.asarray(d, float)) ** 2) / L


def beta_mm(f_hz, eeff):
    return 2 * math.pi * f_hz * math.sqrt(eeff) / C0 / 1e3


def fence_transmission(xy, gnd, lam, rmin=0.35, rmax=3.0):
    """(amplitude transmission, n fence vias, widest gap) of the ring of ground vias `gnd` (n, 3: x, y, drill)
    round a via at xy for a wave of wavelength lam (mm). Each angular gap between fence vias is a below-cut-off
    channel (E parallel to the posts: cut-off at width lambda/2) one drill long, entered with an aperture factor
    2g/lambda; gaps weighted by the angle they subtend. No fence: 1."""
    if len(gnd) == 0:
        return 1.0, 0, math.inf
    d = gnd[:, :2] - xy
    r = np.hypot(d[:, 0], d[:, 1])
    sel = (r > rmin) & (r < rmax)
    if sel.sum() < 2:
        return 1.0, int(sel.sum()), math.inf
    a = np.arctan2(d[sel, 1], d[sel, 0])
    o = np.argsort(a)
    ang, rr = a[o], r[sel][o]
    drill = float(np.median(gnd[sel, 2]))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    t2, gmax = 0.0, 0.0
    for k, ga in enumerate(gaps):
        R = 0.5 * (rr[k] + rr[(k + 1) % len(rr)])
        g = max(ga * R - drill, 1e-3)
        gmax = max(gmax, g)
        if g >= lam / 2:
            T = 1.0
        else:
            alpha = 2 * np.pi / lam * np.sqrt((lam / (2 * g)) ** 2 - 1)
            T = min(1.0, 2 * g / lam) * np.exp(-alpha * drill)
        t2 += ga / (2 * np.pi) * T ** 2
    return float(np.sqrt(t2)), int(sel.sum()), float(gmax)


def cavity_coupling(r, f_hz, eps, d):
    """|Z21| / 2 Z0 (Z0 = 50) between two vias r mm apart through a parallel-plate cavity of height d mm."""
    from scipy.special import hankel2

    k = 2 * math.pi * f_hz * math.sqrt(eps) / C0 / 1e3
    eta = 376.73 / math.sqrt(eps)
    return eta * k * d / 4 * abs(hankel2(0, k * r)) / (2 * 50.0)


def _resample(P, step):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    L = s[-1]
    n = max(2, int(L / step) + 1)
    si = np.linspace(0, L, n)
    Q = np.c_[np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])]
    return Q, (L / (n - 1) if n > 1 else 0.0)


def sample(chains, cfg, corner):
    """Per chain: samples dict(layer, xy, level, stage, dir, ds, s) and the vias / parts with their levels."""
    f = corner["f"]
    out = {}
    for name, items in chains.items():
        lvl, stage, s0 = 0.0, 0, 0.0
        lay, xy, lev, stg, dr, dss, ss = [], [], [], [], [], [], []
        vias, parts, names = [], [], []
        for it in items:
            if it[0] == "part":
                g = part_gain(cfg, it[1], f, corner["t"], corner["states"])
                lvl += g
                act = cfg["parts"][it[1]].get("active", False)
                if act:
                    stage += 1
                    names.append(it[2])
                parts.append(dict(key=it[1], label=it[2], gain=g, level=lvl, s=s0, active=act,
                                  xy=None if it[3] is None else [float(it[3][0]), float(it[3][1])]))
                continue
            if it[0] == "via":
                lvl -= it[2].get("loss_db", cfg["via_loss_db"])
                vias.append(dict(xy=np.asarray(it[1], float), level=lvl, stage=stage, s=s0, opts=it[2]))
                continue
            L = cfg["layers"][it[1]]
            loss = L["loss_db_per_mm"] * math.sqrt(f / L["loss_f_ghz"])
            P, ds = _resample(np.asarray(it[2], float), cfg["step"])
            d = np.gradient(P, axis=0)
            d /= np.linalg.norm(d, axis=1)[:, None] + 1e-12
            k = np.arange(len(P))
            lay += [it[1]] * len(P)
            xy.append(P)
            lev.append(lvl - loss * ds * k)
            stg += [stage] * len(P)
            dr.append(d)
            dss += [ds] * len(P)
            ss.append(s0 + ds * k)
            lvl -= loss * ds * (len(P) - 1)
            s0 += ds * (len(P) - 1)
        out[name] = dict(layer=np.array(lay), xy=np.vstack(xy), level=np.concatenate(lev), stage=np.array(stg),
                         dir=np.vstack(dr), ds=np.array(dss), s=np.concatenate(ss), vias=vias, parts=parts,
                         stage_names=names, end_level=lvl)
    return out


def stage_name(names, k):
    if not names:
        return "path"
    if k == 0:
        return f"before {names[0]}"
    if k >= len(names):
        return f"after {names[-1]}"
    return f"{names[k - 1]} - {names[k]}"


def line_line_loops(S, cfg, corner):
    from scipy.spatial import cKDTree

    f = corner["f"] * 1e9
    rows = []
    for ch, smp in S.items():
        for layer, L in cfg["layers"].items():
            idx = np.nonzero(smp["layer"] == layer)[0]
            if len(idx) < 2:
                continue
            XY, lev, stg, D, ds = (smp["xy"][idx], smp["level"][idx], smp["stage"][idx], smp["dir"][idx],
                                   smp["ds"][idx])
            beta = beta_mm(f, L["eeff"])
            kL = lambda d: coupling_kL(d, L["h"], L["z0"], L["eeff"])  # noqa: E731
            kc = L["kc_ratio"]
            tree = cKDTree(XY)
            run, runs = None, []
            jump = cfg["run_gap"] / cfg["step"]
            for i in range(len(XY)):
                best = None
                for j in tree.query_ball_point(XY[i], cfg["dmax"]):
                    if stg[j] >= stg[i] or lev[i] - lev[j] <= 0:
                        continue
                    dd = float(np.hypot(*(XY[i] - XY[j])))
                    if dd < cfg["dmin"] or abs(D[i] @ D[j]) < cfg["parallel_cos"]:
                        continue
                    lg = lev[i] - lev[j] + 20 * math.log10(kL(dd))
                    if best is None or lg > best[0]:
                        best = (lg, j, dd)
                if best is None:
                    if run:
                        runs.append(run)
                    run = None
                    continue
                _, j, dd = best
                k = float(kL(dd))
                if run and abs(j - run["jlast"]) <= jump and i - run["ilast"] <= 2:
                    run["sum"] += k * ds[i]
                    run["kmax"] = max(run["kmax"], k)
                    run["dmin"] = min(run["dmin"], dd)
                    run["dg"] = max(run["dg"], lev[i] - lev[j])
                    run["n"] += 1
                    run["jlast"], run["ilast"] = j, i
                    run["agg"].append(int(idx[i]))
                    run["vic"].append(int(idx[j]))
                else:
                    if run:
                        runs.append(run)
                    run = dict(sum=k * ds[i], kmax=k, dmin=dd, dg=lev[i] - lev[j], n=1, jlast=j, ilast=i,
                               agg=[int(idx[i])], vic=[int(idx[j])], stages=(int(stg[j]), int(stg[i])))
            if run:
                runs.append(run)
            for r in runs:
                kb = (1 + kc) / 4 * r["kmax"]
                kf = (1 - kc) / 2 * beta * r["sum"]
                c = 20 * math.log10(max(kb, kf)) + L["coupling_offset_db"]
                fwd = kf > kb
                rows.append(dict(kind="line-line", ch=ch, layer=layer, loop=r["dg"] + c, dg=r["dg"], c=c,
                                 stages=r["stages"], agg=sorted(set(r["agg"])), vic=sorted(set(r["vic"])),
                                 a_xy=smp["xy"][r["agg"][0]].tolist(), v_xy=smp["xy"][r["vic"][0]].tolist(),
                                 note=f"{'forward' if fwd else 'backward'}, {r['n'] * cfg['step']:.1f} mm coupled, "
                                      f">= {r['dmin']:.2f} mm apart"))
    return rows


def via_loops(S, cfg, corner, fences):
    """Via radiation into upstream lines and via-to-via coupling through plane-pair cavities."""
    f = corner["f"] * 1e9
    w = 2 * math.pi * f
    rows = []
    seen = {}
    for ch, smp in S.items():
        for V in smp["vias"]:
            key = tuple(np.round(V["xy"], 3))
            seen.setdefault(key, []).append((ch, V))
    for key, users in seen.items():
        xy = users[0][1]["xy"]
        opts = users[0][1]["opts"]
        for ch, V in users:
            smp = S[ch]
            # (1) radiation from the vertical current into upstream lines of each layer it passes
            for layer in opts.get("radiates_into", list(cfg["layers"])):
                L = cfg["layers"][layer]
                sel = (smp["layer"] == layer) & (smp["stage"] < V["stage"]) & (smp["level"] < V["level"])
                gi = np.nonzero(sel)[0]
                if not len(gi):
                    continue
                d = smp["xy"][gi] - xy
                r = np.hypot(d[:, 0], d[:, 1])
                ok = (r > cfg["rvia_min"]) & (r < cfg["rvia"])
                if not ok.any():
                    continue
                gi, d, r = gi[ok], d[ok], r[ok]
                beta = beta_mm(f, L["eeff"])
                lam = C0 / f / math.sqrt(L["eps"]) * 1e3
                cos = np.abs(d[:, 0] * smp["dir"][gi, 0] + d[:, 1] * smp["dir"][gi, 1]) / r
                rm = r * 1e-3
                H = (2 * L["via_height"] * 1e-3) / (4 * np.pi) * (1 / rm ** 2 + 1j * beta * 1e3 / rm) * \
                    np.exp(-1j * beta * r)
                dV = w * MU0 * L["h"] * 1e-3 * H * cos * smp["ds"][gi] * 1e-3      # per amp of via current
                s = np.cumsum(smp["ds"][gi])
                coh = max(abs(np.sum(dV * np.exp(-1j * beta * (s[-1] - s)))), abs(np.sum(dV * np.exp(-1j * beta * s))))
                inc = float(np.sum(np.abs(dV)))
                T, nf, gmax = fence_transmission(xy, fences[("layer", layer)], lam, cfg["fence_rmin"], cfg["fence_rmax"])
                C = 2 * coh / (2 * L["z0"]) * T          # x2: electric coupling of the same order (upper bound)
                if C <= 0:
                    continue
                strong = gi[np.abs(dV) > 0.1 * np.abs(dV).max()]
                dg = V["level"] - float(smp["level"][strong].min())
                c = 20 * math.log10(C)
                j = gi[int(np.argmax(np.abs(dV)))]
                rows.append(dict(kind="via radiation", ch=ch, layer=layer, loop=dg + c, dg=dg, c=c,
                                 stages=(int(smp["stage"][j]), V["stage"]), agg=[], vic=strong.tolist(),
                                 a_xy=xy.tolist(), v_xy=smp["xy"][j].tolist(), via=xy.tolist(),
                                 note=f"incoherent {20 * math.log10(max(2 * inc / (2 * L['z0']) * T, 1e-30)):.1f} dB; "
                                      f"nearest line {r.min():.1f} mm; fence {nf} vias, widest gap "
                                      f"{gmax:.2f} mm, T {20 * math.log10(max(T, 1e-30)):.0f} dB"))
            # (2) parallel-plate wave in each cavity the via crosses, to upstream vias of the same chain
            for cav in opts.get("cavities", list(cfg["cavities"])):
                Cv = cfg["cavities"][cav]
                lam = C0 / f / math.sqrt(Cv["eps"]) * 1e3
                Ta = fence_transmission(xy, fences[("cavity", cav)], lam, cfg["fence_rmin"], cfg["fence_rmax"])
                for U in smp["vias"]:
                    if U["stage"] >= V["stage"] or _same(U["xy"], xy) or \
                            cav not in U["opts"].get("cavities", list(cfg["cavities"])):
                        continue
                    r = float(np.hypot(*(U["xy"] - xy)))
                    Tb = fence_transmission(U["xy"], fences[("cavity", cav)], lam, cfg["fence_rmin"], cfg["fence_rmax"])
                    c0 = cavity_coupling(r, f, Cv["eps"], Cv["d"])
                    c = 20 * math.log10(max(c0 * Ta[0] * Tb[0], 1e-30))
                    dg = V["level"] - U["level"]
                    rows.append(dict(kind="via cavity", ch=ch, layer=cav, loop=dg + c, dg=dg, c=c,
                                     stages=(U["stage"], V["stage"]), agg=[], vic=[], a_xy=xy.tolist(),
                                     v_xy=U["xy"].tolist(), via=xy.tolist(), via2=U["xy"].tolist(),
                                     note=f"{r:.1f} mm apart; no fence {20 * math.log10(c0):.1f} dB; fences "
                                          f"{Ta[1]} / {Tb[1]} vias, widest gap {Ta[2]:.2f} / {Tb[2]:.2f} mm"))
    return rows


def assess(chains, cfg, corner, fences):
    """(rows sorted worst first, samples) for one corner."""
    S = sample(chains, cfg, corner)
    rows = line_line_loops(S, cfg, corner) + via_loops(S, cfg, corner, fences)
    rows.sort(key=lambda r: -r["loop"])
    for r in rows:
        names = S[r["ch"]]["stage_names"]
        r["agg_stage"], r["vic_stage"] = stage_name(names, r["stages"][1]), stage_name(names, r["stages"][0])
    return rows, S


def fence_sets(cfg, board):
    """{("layer", name) | ("cavity", name): (n, 3) ground vias}; empty without a board."""
    out = {}
    for kind, d in (("layer", cfg["layers"]), ("cavity", cfg["cavities"])):
        for name, x in d.items():
            dl = x.get("fence_drill_layers")
            if "fence" in x:
                out[(kind, name)] = np.asarray(x["fence"], float).reshape(-1, 3)
            elif board is not None:
                out[(kind, name)] = board.fence(dl)
            else:
                out[(kind, name)] = np.zeros((0, 3))
    return out


def assess_all(chains, cfg, board=None):
    fences = fence_sets(cfg, board)
    print("fence vias: " + ", ".join(f"{k[1]} {len(v)}" for k, v in fences.items()), flush=True)
    return [dict(corner=c, rows=r, samples=S) for c in cfg["corners"] for r, S in [assess(chains, cfg, c, fences)]]


# --------------------------------------------------------------------------- outputs

KINDS = ("line-line", "via radiation", "via cavity")


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def summary(rows):
    out = {}
    for k in KINDS:
        sel = [r["loop"] for r in rows if r["kind"] == k]
        if sel:
            out[k] = dict(worst=max(sel), n=len(sel), above_20=sum(x > -20 for x in sel),
                          above_10=sum(x > -10 for x in sel), above_0=sum(x >= 0 for x in sel))
    return out


def verdict(worst):
    if worst >= 0:
        return "oscillation possible: change the layout"
    if worst > -10:
        return "small margin: EM model of the worst couplings"
    if worst > -20:
        return "no oscillation; visible ripple"
    return "low risk"


def write_markdown(results, cfg, path):
    L = [f"# Oscillation screen: {cfg['name']}\n",
         "Loop gain = level difference (aggressor downstream - victim upstream) + coupling, every loop at its worst "
         "phase. Couplings are closed-form upper bounds (bare microstrip, no side grounds). Thresholds: >= 0 dB "
         "oscillation possible; -10..0 dB EM model needed; -20..-10 dB ripple; < -20 dB low risk.\n",
         "| corner | worst loop [dB] | mechanism | verdict |", "|---|---|---|---|"]
    for R in results:
        w = R["rows"][0] if R["rows"] else None
        L.append(f"| {R['corner']['label']} | {w['loop']:+.1f} | {w['kind']} | {verdict(w['loop'])} |" if w else
                 f"| {R['corner']['label']} | - | - | no loops |")
    for R in results:
        L += ["", f"## {R['corner']['label']}\n"]
        for k, s in summary(R["rows"]).items():
            L.append(f"- **{k}**: worst {s['worst']:+.1f} dB; {s['above_20']} above -20 dB, {s['above_10']} above "
                     f"-10 dB, {s['above_0']} >= 0 dB (of {s['n']})")
        L += ["", "| loop gain [dB] | mechanism | chain | layer | aggressor (downstream) | victim (upstream) | "
              "level diff [dB] | coupling [dB] | notes |", "|---|---|---|---|---|---|---|---|---|"]
        for r in R["rows"][:cfg["top_rows"]]:
            L.append(f"| {r['loop']:+.1f} | {r['kind']} | {r['ch']} | {r['layer']} | {r['agg_stage']} at "
                     f"({r['a_xy'][0]:.1f}, {r['a_xy'][1]:.1f}) | {r['vic_stage']} at ({r['v_xy'][0]:.1f}, "
                     f"{r['v_xy'][1]:.1f}) | {r['dg']:.1f} | {r['c']:.1f} | {r['note']} |")
    Path(path).write_text("\n".join(L) + "\n")
    return path


def _svg_path(g, nd=3):
    import shapely

    parts = []
    for p in shapely.get_parts(g):
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        for ring in [p.exterior, *p.interiors]:
            c = np.round(np.asarray(ring.coords), nd)
            parts.append("M" + "L".join(f"{x:g},{y:g}" for x, y in c) + "Z")
    return "".join(parts)


def display_copper(chains, cfg, board):
    """{layer: svg path data} of the traced nets and display nets within display_band of the paths."""
    if board is None:
        return {}
    import shapely
    from shapely.geometry import LineString

    out = {}
    for layer in cfg["layers"]:
        lines = [LineString(it[2]) for items in chains.values() for it in items
                 if it[0] == "line" and it[1] == layer and len(it[2]) > 1]
        if not lines:
            continue
        zone = shapely.union_all([g.simplify(0.05) for g in lines]).buffer(cfg["display_band"], quad_segs=4)
        nets = {it[3] for items in chains.values() for it in items if it[0] == "line" and it[1] == layer and it[3]}
        feats = board.odb.layer_features(layer)
        nf = board.odb.net_features(layer)
        tree_g = shapely.STRtree([zone])
        geoms = []
        for net in sorted(nets) + list(cfg["display_nets"]):
            for i in nf.get(net, []):
                g = feats[i] if i < len(feats) else None
                if g is None or not len(tree_g.query(g, predicate="intersects")):
                    continue
                geoms.append(g.intersection(zone).simplify(cfg["simplify"]))
        cu = shapely.union_all(geoms).simplify(cfg["simplify"]) if geoms else None
        out[layer] = _svg_path(cu) if cu is not None else ""
        print(f"viewer copper {layer}: {len(geoms)} pieces, {len(out[layer]) / 1e6:.1f} MB", flush=True)
    return out


def page_data(results, cfg, copper):
    chains = list(results[0]["samples"])
    layers = list(cfg["layers"])
    smp0 = results[0]["samples"]
    geo = {ch: dict(xy=np.round(s["xy"], 3).ravel().tolist(), layer=[layers.index(x) for x in s["layer"]],
                    s=np.round(s["s"], 2).tolist(), stage=s["stage"].tolist(),
                    vias=[[round(float(v["xy"][0]), 3), round(float(v["xy"][1]), 3)] for v in s["vias"]],
                    stage_names=s["stage_names"]) for ch, s in smp0.items()}
    corners = []
    for R in results:
        rows = [dict(k=KINDS.index(r["kind"]), ch=chains.index(r["ch"]), layer=r["layer"], loop=round(r["loop"], 2),
                     dg=round(r["dg"], 2), c=round(r["c"], 2), a=r["agg_stage"], v=r["vic_stage"],
                     axy=[round(x, 3) for x in r["a_xy"]], vxy=[round(x, 3) for x in r["v_xy"]], agg=r["agg"],
                     vic=r["vic"], via=r.get("via"), via2=r.get("via2"), note=r["note"]) for r in R["rows"]]
        corners.append(dict(label=R["corner"]["label"], rows=rows,
                            level={ch: np.round(s["level"], 2).tolist() for ch, s in R["samples"].items()},
                            parts={ch: s["parts"] for ch, s in R["samples"].items()},
                            vias={ch: [round(v["level"], 2) for v in s["vias"]] for ch, s in R["samples"].items()}))
    lay = {n: {k: v for k, v in L.items() if k != "fence"} for n, L in cfg["layers"].items()}
    return dict(title=f"Oscillation risk: {cfg['name']}", chains=chains, layers=layers, geo=geo, corners=corners,
                copper=copper, kinds=list(KINDS), layer_info=lay,
                cavities={n: {k: v for k, v in c.items() if k != "fence"} for n, c in cfg["cavities"].items()},
                links=cfg.get("links", []))


def write_html(results, cfg, copper, path):
    try:
        from plotly.offline import get_plotlyjs
        js = "<script>" + get_plotlyjs() + "</script>"
    except ImportError:  # pragma: no cover - plotly not installed
        js = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
    tpl = (Path(__file__).parent / "templates" / "oscillation.html").read_text()
    data = page_data(results, cfg, copper)
    html = tpl.replace("__PLOTLY__", js).replace("__DATA__", json.dumps(data, separators=(",", ":"), default=_np))
    Path(path).write_text(html)
    return path


def write_json(results, path):
    def clean(r):
        return {k: v for k, v in r.items() if k not in ("stages",)}
    out = [dict(corner=R["corner"], summary=summary(R["rows"]), rows=[clean(r) for r in R["rows"]]) for R in results]
    Path(path).write_text(json.dumps(out, indent=1, default=_np))
    return path


def run(config_path):
    cfg = load(config_path)
    chains, board = trace(cfg)
    results = assess_all(chains, cfg, board)
    out = cfg.out
    copper = display_copper(chains, cfg, board)
    paths = [write_json(results, out / "oscillation.json"), write_markdown(results, cfg, out / "oscillation.md"),
             write_html(results, cfg, copper, out / "oscillation.html")]
    for R in results:
        w = R["rows"][0]["loop"] if R["rows"] else -math.inf
        print(f"{R['corner']['label']}: worst loop {w:+.1f} dB ({verdict(w)})")
    for p in paths:
        print("wrote", p)
    return results


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m mom25d.oscillation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="oscillation config (JSON)")
    a = ap.parse_args(argv)
    run(a.config)


if __name__ == "__main__":
    main()
