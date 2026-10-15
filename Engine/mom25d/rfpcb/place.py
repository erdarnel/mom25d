"""mom25d layouts of section / merged / complete models: thin sheet with the edge expansion, vias (optionally to an
inner plane), gap ports (drawn, or to separate stitched pads), GCPW feeds, optional inner plane with the DXF voids as
openings. Board mm in, metres to the engine (origin shifted to the section)."""
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from . import dxfio
from .engine import mom25d


def dirname(u):
    for k, v in {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}.items():
        if np.allclose(u, v, atol=1e-9):
            return k
    return tuple(u)


class Placer:
    def __init__(self, board, plane=None):
        self.b, self.cfg, self.M = board, board.cfg, board.cfg.m
        self.st = board.stack
        self.plane = plane if plane is not None else (self.M.get("plane") or {}).get("name")
        self.d = self.st.edge_delta
        self._void = None
        self.void_override = None    # local-coordinate void of a reference line (Project.ref_void); None: the DXF voids

    # ---------------------------------------------------------------- pieces
    def plane_void(self):
        if self.void_override is not None:
            return self.void_override
        if self._void is None:
            lay = self.cfg["layers"]["plane_voids"].format(plane=self.plane)
            self._void = dxfio.voids(self.cfg.dxf, lay)
        return self._void

    def via_ends(self, x, y, to="gnd"):
        """(bottom, connect) of a via: to the inner plane if it ends there, else to the ground, connected to the plane
        on the way unless it passes through a void. Without a plane every via goes to the ground."""
        if not self.plane:
            return "gnd", ()
        if to == self.plane:
            return self.plane, ()
        return "gnd", (() if self.plane_void().contains(Point(x, y)) else (self.plane,))

    def port_geom(self, ref, u, w, gap):
        """Gap port to a separate ground pad beyond a slot, stitched to the ground by its own vias:
        (pad, slot, bridge start, bridge end, pad vias)."""
        M = self.M
        SLOT, PAD_D, side, bi = M["slot"], M["pad_depth"], M["pad_side"], M["bridge_in"]
        n = np.array([-u[1], u[0]])
        hp = w / 2 + side
        pad = Polygon([ref + hp * n + SLOT * u, ref - hp * n + SLOT * u,
                       ref - hp * n + (SLOT + PAD_D) * u, ref + hp * n + (SLOT + PAD_D) * u])
        h = w / 2 + gap
        slot = Polygon([ref + h * n, ref - h * n, ref - h * n + SLOT * u, ref + h * n + SLOT * u])
        k = int(2 * (hp - 0.1) / 0.16) + 1
        pvias = [ref + t * n + (SLOT + PAD_D / 2) * u for t in np.linspace(-(hp - 0.1), hp - 0.1, k)]
        return pad, slot, ref - bi * u, ref + (SLOT + bi) * u, pvias

    def shared_pad(self, pa, pb):
        """Two pad ports facing each other closer than two pad depths (a 0 ohm part): one stitched pad between them.
        pa, pb: port records (name, p, u, w, g). Returns (pad, vias, gap port a, gap port b)."""
        M = self.M
        SLOT, bi = M["slot"], M["bridge_in"]
        u = (pb[1] - pa[1]) / np.hypot(*(pb[1] - pa[1]))
        n = np.array([-u[1], u[0]])
        hp = max(pa[3], pb[3]) / 2 + M["pad_side"]
        q0, q1 = pa[1] + SLOT * u, pb[1] - SLOT * u
        pad = Polygon([q0 + hp * n, q0 - hp * n, q1 - hp * n, q1 + hp * n])
        k = int(2 * (hp - 0.1) / 0.16) + 1
        mid = 0.5 * (q0 + q1)
        pv = [mid + t * n + dd * u for t in np.linspace(-(hp - 0.1), hp - 0.1, k) for dd in (-0.12, 0.12)]
        return pad, pv, (pa[0], pa[1] - bi * u, q0 + bi * u, pa[3]), (pb[0], pb[1] + bi * u, q1 - bi * u, pb[3])

    def feed_ground(self, gap, band):
        m = mom25d()
        d, M = self.d, self.M
        return m.CoplanarGround(gap=(gap - 2 * d) * 1e-3, width=(band - gap + d) * 1e-3, via_pitch=M["fence_pitch"] * 1e-3,
                                via_drill=M["fence_drill"] * 1e-3, via_pad=M["fence_pad"] * 1e-3,
                                via_connect=(self.plane,) if self.plane else ())

    # ---------------------------------------------------------------- layout
    def place(self, lay, rf_list, gnd, vias, o, gap_ports, feeds=(), pads=(), band=None, sparams=()):
        """Copper, vias, pads, plane openings, gap ports, feeds into `lay` (board mm; o = model origin).
        gap_ports [(name, sig, ref, w)], feeds [(name, ref, u, w, gap)], pads [(polygon, [via xy])]."""
        m = mom25d()
        mm = 1e-3
        M = self.M
        band = M["ground_band"] if band is None else band
        sig = self.st.signal
        S = lambda g: affinity.scale(affinity.translate(g, -o[0], -o[1]), mm, mm, origin=(0, 0))
        P = lambda p: ((p[0] - o[0]) * mm, (p[1] - o[1]) * mm)
        d = self.d
        rfs = [r.buffer(d, join_style="mitre") if d else r for r in rf_list]
        g = gnd.buffer(d, join_style="mitre") if d else gnd
        strips = []
        for name, ref, u, w, gap in feeds:
            p = m.Layout().add_port(name, ref, dirname(u), w + 2 * d, M["feed_length"], layer=sig,
                                    ground=m.CoplanarGround(gap=gap - 2 * d, width=band - gap + d))
            strips += [Polygon(c) for c in p.ground_strips()]
        if strips:
            g = unary_union([g] + strips).buffer(0.015, join_style="mitre").buffer(-0.015, join_style="mitre")
        g = g.difference(unary_union([r.buffer(0.02, join_style="mitre") for r in rfs]))
        g = shapely.make_valid(shapely.set_precision(g, 0.005).simplify(0.01))
        for r in rfs:
            lay.add_polygon(S(r), layer=sig)
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and p.area > 1e-3:
                lay.add_polygon(S(p), layer=sig)
        for x, y, drill, pad, layer, to in vias:
            bot, con = self.via_ends(x, y, to)
            lay.add_via((x - o[0]) * mm, (y - o[1]) * mm, drill=drill * mm, pad=(drill + 0.02) * mm, top=sig, bottom=bot,
                        connect=con)
        for poly, pv in pads:
            lay.add_polygon(S(poly.buffer(d, join_style="mitre") if d else poly), layer=sig)
            for q in pv:
                bot, con = self.via_ends(q[0], q[1])
                lay.add_via(*P(q), drill=M["pad_via"] * mm, pad=(M["pad_via"] + 0.02) * mm, top=sig, bottom=bot, connect=con)
        if self.plane:
            pl = self.M.get("plane") or {}
            lay.add_plane(self.plane)
            allcu = unary_union(rfs + [g] + [poly for poly, _ in pads])
            vd = self.plane_void()
            for name, ref, u, w, gap in feeds:   # a feed carries the line on: the plane is voided under it as under the line
                ref, u = np.asarray(ref, float), np.asarray(u, float)
                n = np.array([-u[1], u[0]]); h = w / 2 + gap + pl.get("void_margin", 0.05)
                L = M["feed_length"] + 1.0
                vd = vd.union(Polygon([ref + h * n, ref - h * n, ref - h * n + L * u, ref + h * n + L * u]))
            ap = vd.intersection(box(*allcu.bounds).buffer(1.0, join_style="mitre"))
            ap = shapely.make_valid(shapely.set_precision(ap, 0.005).simplify(0.01))
            ap = unary_union([q for q in getattr(ap, "geoms", [ap]) if q.geom_type == "Polygon" and q.area > 2e-3])
            if not ap.is_empty:
                lay.add_aperture(self.plane, S(ap), max_cell=pl.get("ap_max", 0.15) * mm, edge_cell=pl.get("ap_edge", 0.04) * mm)
        for name, s0, r0, w in gap_ports:
            lay.add_gap_port(name, P(s0), P(r0), (w + 2 * d) * mm, layer=sig, z_ref=50.0)
        for name, f, prs, w in sparams:
            lay.add_sparams(name, str(f), [(P(a), P(b)) for a, b in prs], (w + 2 * d) * mm, layer=sig)
        for name, ref, u, w, gap in feeds:
            lay.add_port(name, P(ref), dirname(u), (w + 2 * d) * mm, M["feed_length"] * mm, layer=sig,
                         ground=self.feed_ground(gap, band))

    def build(self, sec, band=None, share_pads=True):
        """Layout + mesh of a section / merged / complete model. Pad ports closer than two pad depths (+ slot) to the
        next port of the group share one stitched pad."""
        m = mom25d()
        M = self.M
        bi = M["bridge_in"]
        pads, gp, fd = [], [], []
        ports = sec["ports"]
        kinds = sec["kinds"]
        shared = set()
        if share_pads:
            lim = 2 * (M["slot"] + M["pad_depth"])
            for i in range(1, len(ports) - 1, 2):
                a, b = ports[i], ports[i + 1]
                if kinds[i] == kinds[i + 1] == "pad" and np.hypot(*(b[1] - a[1])) < lim:
                    pad, pv, ga, gb = self.shared_pad(a, b)
                    pads.append((pad, pv)); gp += [ga, gb]
                    shared |= {a[0], b[0]}
        for (name, ref, u, w, gap), kind in zip(ports, kinds):
            if name in shared:
                continue
            if kind == "feed":
                fd.append((name, ref, u, w, gap))
            elif kind == "gap":
                g = sec["gapports"][name]
                gu = np.asarray(g["u"])
                gp.append((name, np.asarray(g["line"]) - bi * gu, np.asarray(g["gnd"]) + bi * gu, g["w"]))
            else:
                pad, slot, s0, r0, pv = self.port_geom(ref, u, w, gap)
                pads.append((pad, pv)); gp.append((name, s0, r0, w))
        gnd = sec["gnd"].difference(sec["rf"].buffer(M["snap_rf"]))
        lay = m.Layout()
        self.place(lay, list(getattr(sec["rf"], "geoms", [sec["rf"]])), gnd, sec["vias"], sec["origin"], gp, feeds=fd,
                   pads=pads, band=band)
        return lay, lay.mesh(M["max_cell"] * 1e-3, edge_cell=M["edge_cell"] * 1e-3, kind="tri")
