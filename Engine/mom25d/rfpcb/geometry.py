"""Section geometry from the DXF: the line sections of the chain cut at their port lines, their ground (band, nearest-RF
ownership, islands, gap-port lands), vias and ports; merged groups (lines modelled as one, e.g. both sides of a part
with a grounded paddle) and the complete model (all lines merged).

A port record: (name, p, u, w, g) -- p the reference point on the port line (PORTS line centre / gap-port line end),
u the unit vector pointing OUT of the section, w the line width there, g the coplanar gap there. Board mm.
"""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import unary_union

from . import dxfio
from .stackup import Stack


def _beyond(p, u, half, far=20.0):
    """Rectangle beyond the plane through p (normal u, pointing out of the section), within `half` across."""
    n = np.array([-u[1], u[0]])
    return Polygon([p + half * n, p + half * n + far * u, p - half * n + far * u, p - half * n])


def _clean(g):
    g = shapely.set_precision(g.buffer(0.002, join_style="mitre").buffer(-0.002, join_style="mitre"), 0.0005)
    return g.simplify(0.002)


def ownership(rfs, step=0.05, far=50.0):
    """Region closest to each RF piece: Voronoi cells of points sampled along every piece's outline, merged per piece."""
    pts, lab = [], []
    for k, r in enumerate(rfs):
        for ring in [r.exterior] + list(r.interiors):
            n = max(8, int(ring.length / step))
            for t in np.linspace(0, ring.length, n, endpoint=False):
                q = ring.interpolate(t)
                pts.append((q.x, q.y)); lab.append(k)
    mp = MultiPoint(pts)
    x0, y0, x1, y1 = mp.bounds
    cells = list(shapely.voronoi_polygons(mp, extend_to=box(x0 - far, y0 - far, x1 + far, y1 + far), ordered=True).geoms)
    return [unary_union([c for c, l in zip(cells, lab) if l == k]).buffer(1e-6) for k in range(len(rfs))]


def cross_cut(metal, p, d, mid=2.0):
    """Cut across the direction d at p: (width of the copper holding p, gap left, gap right), gaps capped at 1 mm;
    None if p is not on copper."""
    n = np.array([-d[1], d[0]])
    ln = LineString([tuple(p - mid * n), tuple(p + mid * n)])
    x = ln.intersection(metal)
    ch = sorted(tuple(sorted((ln.project(Point(g.coords[0])), ln.project(Point(g.coords[-1])))))
                for g in getattr(x, "geoms", [x]) if g.geom_type == "LineString" and not g.is_empty)
    hit = [c for c in ch if c[0] - 1e-6 <= mid <= c[1] + 1e-6]
    if not hit:
        return None
    w0, w1 = hit[0]
    left = max([c[1] for c in ch if c[1] < w0 - 1e-6], default=None)
    right = min([c[0] for c in ch if c[0] > w1 + 1e-6], default=None)
    gl = w0 - left if left is not None else 1.0
    gr = right - w1 if right is not None else 1.0
    return w1 - w0, min(gl, 1.0), min(gr, 1.0)


def _near(a, b, tol=1e-6):
    """a covers / touches b within tol (a buffer(1e-6) test fails on geometries carrying a precision grid)."""
    if a.is_empty or b.is_empty:
        return False
    return a.distance(b) <= tol


def r5(v):
    return round(v / 0.005) * 0.005


class Board:
    """The board of a config: copper, vias, labelled ports, and the sections of the chain."""

    def __init__(self, cfg):
        self.cfg = cfg
        L = cfg["layers"]
        self.stack = Stack(cfg)
        self.metal = dxfio.painted_metal(cfg.dxf, L["metal"], L["not_metal"])
        self.via_layers = self.stack.via_layers(cfg.get("vias"))
        self.vias = dxfio.vias(cfg.dxf, self.via_layers)
        self.gaps = dxfio.gap_ports(cfg.dxf, L["gap_ports"], self.metal)
        self.plines = dxfio.port_lines(cfg.dxf, L["ports"])
        self.lines = cfg.lines()
        self.parts = cfg.parts()
        for e in self.lines:
            for lab in (e["from"], e["to"]):
                if lab not in self.gaps and lab not in self.plines:
                    raise ValueError(f"port {lab!r} (line {e['line']}) is neither a {L['gap_ports']} nor a "
                                     f"{L['ports']} label in the DXF")

    # ---------------------------------------------------------------- ports
    def kind(self, label):
        k = self.cfg["ports"].get(label, {}).get("kind")
        if k:
            return k
        return "gap" if label in self.gaps else "pad"

    def _point(self, label):
        return np.asarray(self.gaps[label]["line"] if label in self.gaps else self.plines[label]["c"], float)

    def port_frame(self, label, other):
        """(p, u, w) of a port: u points out of the section whose other end is `other` (a point)."""
        if label in self.gaps:
            g = self.gaps[label]
            return np.asarray(g["line"], float), np.asarray(g["u"], float), g["w"]
        pl = self.plines[label]
        t = np.asarray(pl["t"])
        n = np.array([-t[1], t[0]])
        p = np.asarray(pl["c"])
        ov = self.cfg["ports"].get(label, {}).get("out")
        if ov:
            u = np.asarray({"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[ov], float)
        else:
            u = n if np.dot(n, np.asarray(other) - p) < 0 else -n
        return p, u, pl["w"]

    def port_gap(self, p, u, step=0.01):
        """Coplanar gap at a port: cut across u, step/2 inside the section, the mean of both gaps rounded to 5 um."""
        c = cross_cut(self.metal, p - 0.5 * step * u, u)
        if c is None:
            return None
        return r5((c[1] + c[2]) / 2)

    # ---------------------------------------------------------------- sections
    def sections(self, merge=None, band=None):
        """[{name, rf, gnd, vias, ports, gapports, origin, members}] per line (or merged group), board mm."""
        cfg, M = self.cfg, self.cfg.m
        merge = M["merge"] if merge is None else merge
        band = M["ground_band"] if band is None else band
        metal, allvias = self.metal, self.vias
        names = [e["line"] for e in self.lines]
        pre = []
        for e in self.lines:
            pa0, pb0 = self._point(e["from"]), self._point(e["to"])
            pa, ua, wa = self.port_frame(e["from"], pb0)
            pb, ub, wb = self.port_frame(e["to"], pa0)
            cuts = unary_union([_beyond(pa, ua, M["rf_cut_half"]), _beyond(pb, ub, M["rf_cut_half"])])
            cu = metal.difference(cuts)
            pieces = [p for p in getattr(cu, "geoms", [cu]) if p.geom_type == "Polygon"]
            qa, qb = Point(*(pa - 0.02 * ua)), Point(*(pb - 0.02 * ub))
            cand = [p for p in pieces if _near(p, qa) and _near(p, qb)]
            if not cand:
                raise ValueError(f"line {e['line']}: no copper piece joins its ports {e['from']} and {e['to']}")
            rf = _clean(cand[0])
            ga = cfg["ports"].get(e["from"], {}).get("gap") or self.port_gap(pa, ua, cfg["circuit"]["step"])
            gb = cfg["ports"].get(e["to"], {}).get("gap") or self.port_gap(pb, ub, cfg["circuit"]["step"])
            pre.append(dict(pa=pa, pb=pb, ua=ua, ub=ub, wa=wa, wb=wb, ga=ga, gb=gb, cuts=cuts, rf=rf,
                            la=e["from"], lb=e["to"]))
        own = ownership([q["rf"] for q in pre]) if M["ownership"] else [box(-1e4, -1e4, 1e4, 1e4)] * len(pre)
        allrf = unary_union([q["rf"] for q in pre])
        groups, done = [], set()
        for k, nm in enumerate(names):
            if k in done:
                continue
            grp = next(([names.index(x) for x in g] for g in merge if nm in g), [k])
            grp = sorted(grp)
            done.update(grp); groups.append(grp)
        out = []
        for grp in groups:
            ends = []
            for m in grp:
                q = pre[m]
                ends += [(q["pa"], q["ua"], q["wa"], q["ga"], q["la"]), (q["pb"], q["ub"], q["wb"], q["gb"], q["lb"])]
            pnames = [f"P{i + 1}" for i in range(len(ends))]
            cuts_v = unary_union([_beyond(ends[0][0], ends[0][1], M["rf_cut_half"]),
                                  _beyond(ends[-1][0], ends[-1][1], M["rf_cut_half"])])
            rf = unary_union([pre[m]["rf"] for m in grp])
            own_k = unary_union([own[m] for m in grp])
            wmax = max(e[2] for e in ends)
            cu = metal.difference(allrf.buffer(1e-4)).difference(Polygon())   # (op order kept: bit-identical snapping)
            gnd = cu.intersection(rf.buffer(band, join_style="mitre")).intersection(own_k)
            gnd = gnd.buffer(-0.02, join_style="mitre").buffer(0.02, join_style="mitre").simplify(0.005)
            gnd = unary_union([_clean(p) for p in getattr(gnd, "geoms", [gnd]) if p.geom_type == "Polygon" and p.area > 1e-3])
            if M["drop_islands"]:
                gnd = unary_union([p for p in getattr(gnd, "geoms", [gnd])
                                   if any(_near(p, Point(v[0], v[1])) for v in allvias)])
            gpk = {}
            for nm, (pp, uu, _w, _g, lab) in zip(pnames, ends):
                if self.kind(lab) != "gap":
                    continue
                g = self.gaps[lab]
                gu = np.asarray(g["u"]); gn = np.array([-gu[1], gu[0]])
                H = g["w"] / 2 + M["land_half"]
                depth = g["slot"] + M["gap_keep"]
                keep = Polygon([pp + H * gn, pp - H * gn, pp - H * gn + depth * gu, pp + H * gn + depth * gu])
                piece = next((q for q in getattr(metal, "geoms", [metal]) if q.buffer(0.005).contains(Point(g["gnd"]))), None)
                if piece is not None and piece.area < M["land_whole"]:
                    land = piece.difference(rf.buffer(0.05))
                else:
                    land = metal.intersection(keep).difference(rf.buffer(0.05))
                    land = unary_union([q for q in getattr(land, "geoms", [land])
                                        if q.geom_type == "Polygon" and q.buffer(0.005).contains(Point(g["gnd"]))])
                gnd = unary_union([gnd, land])
                gpk[nm] = dict(g, land=land, label=lab)
            lands = unary_union([g["land"] for g in gpk.values()]) if gpk else Polygon()
            vias = [v for v in allvias if gnd.contains(Point(v[0], v[1]))
                    and (lands.contains(Point(v[0], v[1])) or not Point(v[0], v[1]).buffer(v[3] / 2).intersects(cuts_v))]
            if M["drop_islands"]:
                keep = [p for p in getattr(gnd, "geoms", [gnd]) if any(_near(p, Point(v[0], v[1])) for v in vias)]
                near = rf.buffer(wmax + 0.15)
                keep = [p for p in keep if p.intersects(near) or _near(p, lands)]
                gnd = unary_union(keep)
                vias = [v for v in vias if gnd.contains(Point(v[0], v[1]))]
            gnd = unary_union([gnd] + [Point(v[0], v[1]).buffer(v[3] / 2, quad_segs=8) for v in vias])
            gnd = shapely.set_precision(gnd, 0.01).simplify(0.02)
            out.append(dict(name="_".join(names[m] for m in grp), rf=rf, gnd=gnd, vias=vias,
                            ports=[(nm, e[0], e[1], e[2], e[3]) for nm, e in zip(pnames, ends)],
                            labels=[e[4] for e in ends], kinds=[self.kind(e[4]) for e in ends],
                            gapports=gpk, origin=0.5 * (ends[0][0] + ends[-1][0]), members=[names[m] for m in grp]))
        return out

    def complete(self):
        """All lines as one model, ground band model.complete_band."""
        return self.sections(merge=[[e["line"] for e in self.lines]], band=self.cfg.m["complete_band"])[0]

    # ---------------------------------------------------------------- centreline (circuit model)
    def centreline(self, line, h=0.01):
        """Centreline of a line section: the shortest path between its two port points along the medial axis of its RF
        copper (Voronoi edges of the densely sampled outline that lie inside the copper; follows tapers and bends,
        ignores side branches into pads). Returns (points (n, 2) every ~h, unit tangents (n, 2))."""
        import heapq
        sec = next(s for s in self.sections(merge=[]) if s["name"] == line)
        rf = shapely.set_precision(sec["rf"], 0.0)          # drop the precision grid (buffers snap to it otherwise)
        (_, pa, ua, _, _), (_, pb, ub, _, _) = sec["ports"]
        pts = []
        for ring in [rf.exterior] + list(rf.interiors):
            n = max(16, int(ring.length / (h / 2)))
            pts += [ring.interpolate(t).coords[0] for t in np.linspace(0, ring.length, n, endpoint=False)]
        edges = shapely.voronoi_polygons(MultiPoint(pts), only_edges=True)
        inner = rf.buffer(-1e-4)
        key = lambda q: (round(q[0], 6), round(q[1], 6))
        G = {}
        for e in getattr(edges, "geoms", [edges]):
            c = list(e.coords)
            for q0, q1 in zip(c[:-1], c[1:]):
                if inner.contains(Point(q0)) and inner.contains(Point(q1)):
                    k0, k1 = key(q0), key(q1)
                    L = math.dist(q0, q1)
                    G.setdefault(k0, []).append((k1, L)); G.setdefault(k1, []).append((k0, L))
        if not G:
            raise ValueError(f"line {line}: no medial axis found")
        nodes = np.array(list(G))
        start = tuple(nodes[np.argmin(np.hypot(*(nodes - (pa - 0.5 * h * ua)).T))])
        goal = tuple(nodes[np.argmin(np.hypot(*(nodes - (pb - 0.5 * h * ub)).T))])
        dist, prev, pq = {start: 0.0}, {}, [(0.0, start)]
        while pq:
            dcur, k = heapq.heappop(pq)
            if k == goal:
                break
            if dcur > dist.get(k, 1e18):
                continue
            for k2, L in G[k]:
                nd = dcur + L
                if nd < dist.get(k2, 1e18):
                    dist[k2], prev[k2] = nd, k
                    heapq.heappush(pq, (nd, k2))
        if goal not in dist:
            raise ValueError(f"line {line}: ports not connected along the copper's medial axis")
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path = path[::-1]
        # the medial axis forks to the corners near a line end: drop the trunk within a line width of each port point
        (_, _, _, wa, _), (_, _, _, wb, _) = sec["ports"]
        i0 = next((i for i, q in enumerate(path) if math.dist(q, pa) > 0.6 * wa), 0)
        i1 = next((i for i in range(len(path) - 1, -1, -1) if math.dist(path[i], pb) > 0.6 * wb), len(path) - 1)
        path = [tuple(pa)] + (path[i0:i1 + 1] if i1 >= i0 else []) + [tuple(pb)]
        cl = LineString(path).simplify(h / 4)
        n = max(2, int(math.ceil(cl.length / h)) + 1)
        P = np.array([cl.interpolate(t).coords[0] for t in np.linspace(0, cl.length, n)])
        T = np.gradient(P, axis=0)
        T /= np.maximum(np.hypot(*T.T), 1e-12)[:, None]
        return P, T

    def segments(self):
        """{line: [(length, width, gap)]} uniform segments for the circuit model. Cross-cuts every circuit.step along
        the centreline (auto, or the circuit.path override); cuts near bends take the nearest straight value."""
        C = self.cfg["circuit"]
        step, keep = C["step"], C["bend_keep"]
        out = {}
        if C.get("path"):
            return self._segments_path(C["path"], step, keep)
        for e in self.lines:
            P, Tn = self.centreline(e["line"], h=step)
            seglen = np.hypot(*np.diff(P, axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seglen)])
            Ltot = s[-1]
            ang = np.unwrap(np.arctan2(Tn[:, 1], Tn[:, 0]))
            bend = np.zeros(len(P), bool)
            for i in range(len(P)):
                lo, hi = np.searchsorted(s, s[i] - keep), np.searchsorted(s, s[i] + keep)
                hi = min(hi, len(P) - 1)
                if abs(ang[hi] - ang[lo]) > np.radians(10):
                    bend[i] = True
            ss = np.arange(step / 2, Ltot, step)
            vals = []
            for sv in ss:
                i = min(np.searchsorted(s, sv), len(P) - 1)
                if bend[i]:
                    vals.append(None); continue
                t = (sv - s[i - 1]) / max(seglen[i - 1], 1e-12) if i > 0 else 0.0
                q = P[i - 1] + t * (P[i] - P[i - 1]) if i > 0 else P[0]
                vals.append(cross_cut(self.metal, q, Tn[i]))
            out[e["line"]] = self._merge_vals(ss, vals, Ltot, step, [np.searchsorted(s, sv) for sv in ss], bend)
        return out

    @staticmethod
    def _merge_vals(ss, vals, Ltot, step, idx, bend):
        good = [i for i, v in enumerate(vals) if v is not None]
        if not good:
            raise ValueError("no valid cross-cuts on a line")
        side = lambda j: int(np.sum(np.diff(bend[: min(idx[j], len(bend) - 1) + 1].astype(int)) == -1))
        for i, v in enumerate(vals):
            if v is None:
                same = [g for g in good if side(g) == side(i)] or good
                vals[i] = vals[min(same, key=lambda g: abs(g - i))]
        segs = []
        for w, gl, gr in vals:
            w, g = r5(w), r5((gl + gr) / 2)
            if segs and abs(segs[-1][1] - w) < 1e-9 and abs(segs[-1][2] - g) < 1e-9:
                segs[-1][0] += step
            else:
                segs.append([step, w, g])
        scale = Ltot / sum(x[0] for x in segs)
        return [(x[0] * scale, x[1], x[2]) for x in segs]

    def _segments_path(self, path, step, keep):
        """Centreline override: a polyline of points (corners rounded at bend_r); port positions are projected on it."""
        pts = [tuple(p) for p in path["points"]]
        R = path.get("bend_r", 0.0)
        legs, s0 = [], 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            L = math.dist(a, b)
            legs.append((s0, s0 + L, a, b)); s0 += L
        corner_s = [legs[i][1] for i in range(len(legs) - 1)]

        def s_of(p):
            best, s0 = None, 0.0
            for sa, sb, a, b in legs:
                L = sb - sa
                t = max(0, min(1, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / L ** 2))
                q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                d = math.dist(p, q)
                if best is None or d < best[0]:
                    best = (d, sa + t * L)
            return best[1]

        def cut(s):
            for sa, sb, a, b in legs:
                if sa - 1e-9 <= s <= sb + 1e-9:
                    t = (s - sa) / (sb - sa)
                    p = np.array([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
                    u = np.array([(b[0] - a[0]) / (sb - sa), (b[1] - a[1]) / (sb - sa)])
                    return cross_cut(self.metal, p, u)
            return None
        out = {}
        for e in self.lines:
            sa, sb = sorted((s_of(self._point(e["from"])), s_of(self._point(e["to"]))))
            nb = sum(sa < c < sb for c in corner_s)
            Lc = sb - sa - nb * (2 - math.pi / 2) * R
            ss = np.arange(sa + step / 2, sb, step)
            vals = [None if any(abs(s - c) < keep for c in corner_s) else cut(s) for s in ss]
            good = [i for i, v in enumerate(vals) if v is not None]
            for i, v in enumerate(vals):
                if v is None:
                    side = lambda j: sum(ss[j] > c for c in corner_s)
                    same = [g for g in good if side(g) == side(i)] or good
                    vals[i] = vals[min(same, key=lambda g: abs(g - i))]
            segs = []
            for w, gl, gr in vals:
                w, g = r5(w), r5((gl + gr) / 2)
                if segs and abs(segs[-1][1] - w) < 1e-9 and abs(segs[-1][2] - g) < 1e-9:
                    segs[-1][0] += step
                else:
                    segs.append([step, w, g])
            scale = Lc / sum(x[0] for x in segs)
            out[e["line"]] = [(x[0] * scale, x[1], x[2]) for x in segs]
        return out
