"""Unstructured triangular meshing of a Layout (constrained Delaunay, via
Shewchuk's Triangle: `pip install triangle`; needs shapely).

Per metal layer:
- every port feed line gets a structured strip in its own coordinates (s
  along the feed, t across; any direction): lines at the copper edges,
  `edge_cell` inside them and in between; transverse cuts at most `s_edge`
  apart, incl. the reference plane, gap and open end. Cells are split into
  triangles with alternating diagonals like the tensor grid. Port cuts are
  therefore mesh edges and `PortModel` works unchanged.
- every other copper edge gets a constraint line `edge_cell` inside it
  (edge-current singularity). Edges and these lines are divided every
  s_edge = min(max_cell, max_aspect * edge_cell).
- the thin edge band between them is triangulated from those points only
  (no refinement), so its cells keep the aspect ratio <= ~max_aspect along
  straight edges -- like the tensor grid's edge strips, but bounded.
- the interior (exact polygons: rectangles, shapely polygons, pours, round
  pads; via holes as constraint circles) is refined to a smallest angle of
  `min_angle` and a largest area of max_cell^2 / 2.
- all parts share their boundary nodes (no Steiner points on segments), so
  the mesh is conforming; a check refuses free edges inside copper.
"""

import numpy as np
from scipy.spatial import cKDTree


def _fill(lines, step):
    """Sorted unique coordinates with every gap subdivided to <= step."""
    lines = np.unique(np.asarray(lines, float))
    out = [lines[:1]]
    for a, b in zip(lines[:-1], lines[1:]):
        n = max(int(np.ceil((b - a) / step - 1e-9)), 1)
        out.append(a + (b - a) * np.arange(1, n + 1) / n)
    return np.concatenate(out)


def _dedupe(vals, keep, d):
    """Values not within d of `keep` or of an earlier kept value."""
    out = list(keep)
    for v in sorted(vals):
        if not out or np.min(np.abs(np.asarray(out) - v)) >= d:
            out.append(v)
    return out[len(keep):]


def _drop_close(lines, keep, d):
    """Drop entries of `lines` closer than d to any of `keep`."""
    keep = np.asarray(keep, float)
    if not keep.size:
        return list(lines)
    return [c for c in lines if np.min(np.abs(keep - c)) >= d]


def _on_feed_boundary(pts, port, tol):
    """Points on the boundary of a port's feed rectangle (any direction)."""
    s, t = port.local(pts)
    W2, L = port.width / 2, port.length
    ends = ((np.abs(s) < tol) | (np.abs(s - L) < tol)) & (np.abs(t) < W2 + tol)
    sides = (np.abs(np.abs(t) - W2) < tol) & (s > -tol) & (s < L + tol)
    return ends | sides


class _Feed:
    """Structured strip of one port feed line."""

    def __init__(self, port, e, s_edge, max_cell, r_max, extra_s=(), extra_t=(), d_merge=0.0):
        self.port = port
        W, L = port.width, port.length
        base = [0.0, port.s_gap, L]
        self.s = _fill(base + _dedupe(extra_s, base, d_merge), s_edge)
        extra_t = _dedupe(extra_t, [-W / 2, W / 2], d_merge)
        t = [-W / 2, W / 2, *extra_t]
        if e and W > 3 * e:
            t += _drop_close([-W / 2 + e, W / 2 - e], extra_t, 0.5 * e)
        step = min(max_cell, r_max * float(np.min(np.diff(self.s))))
        self.t = _fill(t, step)
        S, T = np.meshgrid(self.s, self.t, indexing="ij")
        self.nodes = port.point(S.ravel(), T.ravel())
        ns, nt = len(self.s), len(self.t)
        ii, jj = np.meshgrid(np.arange(ns - 1), np.arange(nt - 1), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        nid = lambda i, j: i * nt + j  # noqa: E731
        p, q, r, u = nid(ii, jj), nid(ii + 1, jj), nid(ii + 1, jj + 1), nid(ii, jj + 1)
        flip = (ii + jj) % 2 == 1
        t1 = np.where(flip[:, None], np.stack([p, q, u], 1), np.stack([p, q, r], 1))
        t2 = np.where(flip[:, None], np.stack([q, r, u], 1), np.stack([p, r, u], 1))
        self.tris = np.concatenate([t1, t2])
        on = np.zeros((ns, nt), bool)  # boundary nodes, shared with the rest
        on[[0, -1], :] = True
        on[:, [0, -1]] = True
        self.boundary = self.nodes[on.ravel()]


def _densify(coords, step, fixed, ports, tol, closed):
    """Points along a polyline: segments on a feed boundary use exactly the
    feed's nodes on them, all others are divided to <= step."""
    c = np.asarray(coords, float)[:, :2]
    pts = []
    for p, q in zip(c[:-1], c[1:]):
        d = q - p
        L = float(np.hypot(*d))
        if L < tol:
            continue
        pts.append(p)
        mid = 0.5 * (p + q)[None]
        if any(_on_feed_boundary(mid, pt, tol)[0] for pt in ports):
            if fixed.size:
                t = (fixed - p) @ d / L**2
                perp = np.abs((fixed[:, 0] - p[0]) * d[1] - (fixed[:, 1] - p[1]) * d[0]) / L
                on = (perp < tol) & (t * L > tol) & ((1 - t) * L > tol)
                pts.extend(fixed[on][np.argsort(t[on])])
            continue
        n = int(np.ceil(L / step - 1e-9))
        pts.extend(p + d * k / n for k in range(1, n))
    if not closed:
        pts.append(c[-1])
    return pts


def _inside_points(geom, d):
    """One point inside every polygon of geom, preferably >= d from its edge."""
    out = []
    for p in getattr(geom, "geoms", [geom]):
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        q = p.buffer(-d)
        q = max(getattr(q, "geoms", [q]), key=lambda g: g.area) if not q.is_empty else p
        out.append(q.representative_point().coords[0])
    return out


def _fixed(x):
    """A positive number in plain decimal notation for Triangle's switches: Triangle reads only digits and '.', so
    exponent notation ('a1.125000e-08') ended the area switch at 1.125 and the rest became other switches -- the
    area bound never applied to SI-unit meshes (decision 37)."""
    s = f"{x:.20f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def _refine_to_size(out, size, amax, min_angle, rounds=6):
    """Refine a Triangle result until every triangle's area is below 0.5 size(centroid)^2 (and amax): Triangle's
    'r' + per-triangle area bounds, no Steiner points on segments (YY), so the edge band's nodes stay shared."""
    import triangle

    for _ in range(rounds):
        V, T = out["vertices"], out["triangles"]
        P = V[T]
        area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                            - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
        target = np.minimum(0.5 * np.asarray(size(P.mean(1)), float) ** 2, amax)
        if np.all(area <= 1.05 * target):
            break
        src = dict(vertices=V, triangles=T, triangle_max_area=target)
        if "segments" in out:
            src["segments"] = out["segments"]
        out = triangle.triangulate(src, f"rpq{min_angle:g}aYYQ")
    return out


def graded_size(fine, fine_cell, max_cell, slope=0.3):
    """Cell size field for `size=`: fine_cell [m] within / at the geometry `fine` (shapely, metres; e.g. the signal
    copper and port lines), growing by `slope` metres per metre of distance, capped at max_cell."""
    import shapely

    def f(pts):
        d = shapely.distance(shapely.points(pts), fine)
        return np.minimum(fine_cell + slope * d, max_cell)
    return f


def triangulate_layout(layout, max_cell, edge_cell=None, min_angle=20.0, max_aspect=5.0, size=None):
    """(nodes, tris, tri_layer, layer_names, port_cuts) for Mesh.

    size : optional graded interior: a callable mapping points (n, 2) [m] to the largest cell size there [m]
           (capped by max_cell), e.g. `graded_size` -- fine near the signal copper, coarse in the far pour."""
    import shapely
    import triangle
    from shapely.geometry import Point, Polygon, box

    if not 0 < min_angle <= 30:
        raise ValueError("min_angle must be in (0, 30] degrees (Triangle may not terminate above)")
    if max_aspect < 1:
        raise ValueError("max_aspect must be >= 1")
    e = edge_cell
    r_max = 1 / np.tan(np.radians(min_angle))
    s_edge = min(max_cell, max_aspect * e) if e else max_cell
    # Copper is snapped to a grid of `snap`: CAD details below it (near-
    # duplicate arc points, micron steps at joins) would force slivers on a
    # boundary that may not be refined. Geometric tests use tol >= the
    # snapping error (<= snap / sqrt 2); feed lines and region vertices
    # closer than d_merge are merged.
    lref = min(e or max_cell, s_edge)
    snap = 0.05 * lref
    x0, y0, x1, y1 = layout.bbox()
    tol = max(1e-9 * max(x1 - x0, y1 - y0), snap)
    d_merge = 0.15 * lref
    amax = 0.5 * max_cell**2
    names = layout.layer_names() or ["top"]
    vias = layout.all_vias()

    nodes, tris, tl, cuts = [], [], [], {}
    off = 0
    for li, name in enumerate(names):
        cu = [box(*r) for r, n in zip(layout.rects, layout.rect_layers) if n == name]
        u = layout._poly_union(name)
        if u is not None:
            cu.append(u)
        cu += [Point(v.xy).buffer(v.pad / 2, quad_segs=8) for v in vias if name in v.pad_layers()]
        copper = shapely.set_precision(shapely.union_all(cu).simplify(snap), snap)
        # keep the snapped coordinates but drop the precision model: with it,
        # every later operation runs on the snap grid (cut vertices off the
        # feed boundary; buffer(-snap) of an oblique strip returned empty)
        copper = shapely.set_precision(copper, 0)
        ports = [p for p in list(layout.ports) + list(getattr(layout, "lumped", [])) if p.layer == name]
        feeds_geom = shapely.union_all([Polygon(p.corners()) for p in ports]) if ports else None
        region = copper
        if feeds_geom is not None:
            # cut the feeds out of the snapped copper (cut vertices lie exactly on
            # the feed boundary), then drop slivers < 2 snap wide along the feeds
            region = copper.difference(feeds_geom)
            region = region.buffer(-snap, join_style="mitre").buffer(snap, join_style="mitre")
        polys = [p for p in getattr(region, "geoms", [region])
                 if p.geom_type == "Polygon" and p.area > tol * max_cell]
        reg = shapely.union_all(polys) if polys else None

        # edge band (between the copper edge and the line edge_cell inside)
        # and interior of the non-feed copper
        offset_lines, band, interior = [], None, reg
        if e and polys:
            inner = copper.buffer(-e, join_style="mitre")
            interior = reg.intersection(inner)
            band = reg.difference(inner)
            lines = inner.boundary.intersection(reg).simplify(0.05 * e)
            offset_lines = [np.asarray(ln.coords) for ln in getattr(lines, "geoms", [lines])
                            if ln.geom_type == "LineString" and ln.length > tol]

        # feed strips, with extra lines where region corners / offset lines meet them
        rv = [c for p in polys for r in [p.exterior, *p.interiors] for c in r.coords]
        rv += [c for ln in offset_lines for c in (ln[0], ln[-1])]
        rv = np.array(rv, float).reshape(-1, 2)
        feeds = []
        for p in ports:
            es, et = [], []
            if rv.size:
                hit = rv[_on_feed_boundary(rv, p, tol)]
                s, t = p.local(hit)
                side = np.abs(np.abs(t) - p.width / 2) < tol
                es = s[side & (s > tol) & (s < p.length - tol)]
                et = t[~side]
            f = _Feed(p, e, s_edge, max_cell, r_max, es, et, d_merge)
            feeds.append(f)
            cuts[p.name] = f.s
        fixed = np.concatenate([f.boundary for f in feeds]) if feeds else np.zeros((0, 2))
        ftree = cKDTree(fixed) if feeds else None

        def snap_to_feeds(c):
            """Move points lying on a feed boundary onto its nearest node."""
            c = np.array(c, float)[:, :2]
            if ftree is None:
                return c
            on = np.zeros(len(c), bool)
            for pt in ports:
                on |= _on_feed_boundary(c, pt, tol)
            if on.any():
                d, k = ftree.query(c[on])
                ok = d < d_merge
                idx = np.nonzero(on)[0][ok]
                c[idx] = fixed[k[ok]]
            return c

        # PSLG of the non-feed copper
        P, S, holes = [], [], []
        drills = []  # via centres inside a drill constraint circle
        drill_geom = []

        def add(pts, closed):
            n0 = len(P)
            P.extend(pts)
            k = np.arange(n0, len(P))
            S.extend(zip(k, np.roll(k, -1)) if closed else zip(k[:-1], k[1:]))

        for p in polys:
            for r in [p.exterior, *p.interiors]:
                add(_densify(snap_to_feeds(r.coords), s_edge, fixed, ports, tol, True), True)
            for r in p.interiors:
                # hole points must lie in empty space: not on copper islands inside the
                # hole, not on feeds / gap bridges (meshed on their own), not on slivers
                # dropped from `polys`; a bridge can split the clearance into several
                # pockets, so one point in each empty piece
                h = Polygon(r).difference(copper if feeds_geom is None else copper.union(feeds_geom))
                for part in getattr(h, "geoms", [h]):
                    if part.geom_type == "Polygon" and part.area > tol * tol:
                        holes.append(part.representative_point().coords[0])
        for ln in offset_lines:
            add(_densify(snap_to_feeds(ln), s_edge, fixed[:0], [], tol, False), False)
        for v in vias:
            c = Point(v.xy)
            if interior is not None and interior.contains(c.buffer(v.drill / 2)):
                # hexagon: a few triangles inside the hole, little grading around it
                a = np.arange(7) * np.pi / 3
                ring = np.c_[v.xy[0] + 0.5 * v.drill * np.cos(a), v.xy[1] + 0.5 * v.drill * np.sin(a)]
                add(_densify(ring, s_edge, fixed[:0], [], tol, True), True)
                drills.append(v.xy)
                drill_geom.append(Polygon(ring).buffer(0.1 * v.drill))

        lay_nodes, lay_tris = [], []
        if P:
            P = np.asarray(P, float)
            grp = _merge(P, tol)  # merge coincident input points
            uniq, inv = np.unique(grp, return_inverse=True)
            S = np.unique(np.sort(inv[np.asarray(S)], 1), axis=0)
            S = S[S[:, 0] != S[:, 1]]
            V = P[uniq]
            # pass 1: edge band from its points only (anisotropic, no refinement);
            # pass 2: interior, refined to min_angle. No Steiner points on any
            # segment (YY), so both passes and the feeds share their nodes.
            # (closed loops inside the interior -- drill circles -- need their own
            # hole point in pass 1)
            cut1 = interior
            if drill_geom and interior is not None:  # hole points must avoid the drills
                cut1 = interior.difference(shapely.union_all(drill_geom))
            passes = [(band, cut1, drills, "pYYQ"),
                      (interior, band, [], f"pq{min_angle:g}a{_fixed(amax)}YYQ")]
            for keep, cut, extra, opts in passes:
                if keep is None or keep.is_empty:
                    continue
                hp = holes + list(extra)
                hp += _inside_points(cut, 0.1 * (e or max_cell)) if cut is not None else []
                pslg = dict(vertices=V, segments=S)
                if hp:
                    pslg["holes"] = np.asarray(hp)
                out = triangle.triangulate(pslg, opts)
                if size is not None and keep is interior and "triangles" in out and len(out["triangles"]):
                    out = _refine_to_size(out, size, amax, min_angle)
                if "triangles" in out and len(out["triangles"]):
                    lay_nodes.append(out["vertices"])
                    lay_tris.append(out["triangles"] + sum(len(n) for n in lay_nodes[:-1]))
        base = sum(len(n) for n in lay_nodes)
        for f in feeds:
            lay_nodes.append(f.nodes)
            lay_tris.append(f.tris + base)
            base += len(f.nodes)
        if not lay_nodes:
            continue
        N = np.concatenate(lay_nodes)
        T = np.concatenate(lay_tris)
        grp = _merge(N, tol)
        uniq, inv = np.unique(grp, return_inverse=True)
        N, T = N[uniq], inv[T]
        T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
        _check_conforming(N, T, copper, tol, name)
        nodes.append(N)
        tris.append(T + off)
        tl.append(np.full(len(T), li))
        off += len(N)
    return (np.concatenate(nodes), np.concatenate(tris), np.concatenate(tl), names, cuts)


def _merge(P, tol):
    """Index of the representative (smallest index) of each point's
    cluster of points closer than tol."""
    grp = np.arange(len(P))
    for i, j in sorted(cKDTree(P).query_pairs(tol)):
        a, b = grp[i], grp[j]
        while grp[a] != a:
            a = grp[a]
        while grp[b] != b:
            b = grp[b]
        grp[max(a, b)] = min(a, b)
    for i in range(len(P)):  # flatten
        r = i
        while grp[r] != r:
            r = grp[r]
        grp[i] = r
    return grp


def _check_conforming(N, T, copper, tol, name):
    """Free edges (one triangle) must lie on the copper boundary, and no
    edge may be shared by more than two triangles (overlapping passes)."""
    import shapely

    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    if np.any(cnt > 2):
        x, y = N[u[cnt > 2][0]].mean(0) * 1e3
        raise ValueError(f"layer {name}: overlapping triangles near ({x:.4g}, {y:.4g}) mm")
    free = u[cnt == 1]
    mid = N[free].mean(1)
    inside = shapely.contains_xy(copper.buffer(-10 * tol), mid[:, 0], mid[:, 1])
    if np.any(inside):
        x, y = mid[np.argmax(inside)] * 1e3
        raise ValueError(f"layer {name}: non-conforming triangulation (hanging node near "
                         f"({x:.4g}, {y:.4g}) mm)")
