"""Mesh morphing: re-use a triangular mesh's connectivity for a slightly changed layout.

For tuning loops (finite differences, space mapping, least squares), re-triangulating every perturbed
layout changes the mesh topology and with it the discretisation error: neighbouring designs differ by
mesh noise (about 1 dB in return loss on a hairpin filter, 2026-10-11), which swamps 5-50 um
steps. Morphing keeps the triangles and moves the nodes with the copper instead, so S varies
smoothly with the geometry.

    mesh1 = layout1.mesh(max_cell, edge_cell, kind="tri", like=mesh0)   # mesh1.morphed: True / False

Requirements (else Layout.mesh falls back to a fresh mesh, morphed = False): kind="tri"; the same
polygons (count, layers, ring structure and vertex count per ring), rectangles, ports (width,
length, direction, layer, ground), vias (count, drill, pad, layers) and thick layers; no pours,
lumped elements, components or apertures. Node displacement:

- nodes near a via (inside its pad) move with the via;
- nodes on a copper edge keep their position along the edge segment (old -> new segment);
- all other nodes take the inverse-distance-squared mean of the displacements of the nearest edge
  nodes of the same copper piece (a rigid shift of a piece moves its interior rigidly).

The result is checked: every triangle keeps its orientation and at least half its smallest angle,
free edges lie on the new copper boundary, and the copper area per layer matches the new layout
within 1 %; otherwise None (fresh mesh).
"""
import numpy as np
from scipy.spatial import cKDTree


def _rings(geom):
    out = []
    for p in getattr(geom, "geoms", [geom]):
        if p.geom_type != "Polygon":
            return None
        out.append([np.asarray(r.coords)[:-1, :2] for r in [p.exterior, *p.interiors]])
    return out


def _compatible(a, b):
    """Shape-level correspondence check of two layouts (see module doc)."""
    if (len(a.polys) != len(b.polys) or list(a.poly_layers) != list(b.poly_layers) or len(a.rects) != len(b.rects)
            or list(a.rect_layers) != list(b.rect_layers) or len(a.ports) != len(b.ports) or len(a.vias) != len(b.vias)
            or dict(a.thick) != dict(b.thick)):
        return False
    if a.pours or b.pours or a.lumped or b.lumped or a.components or b.components or a.apertures or b.apertures:
        return False
    for pa, pb in zip(a.polys, b.polys):
        ra, rb = _rings(pa), _rings(pb)
        if ra is None or rb is None or len(ra) != len(rb):
            return False
        for qa, qb in zip(ra, rb):
            if len(qa) != len(qb) or any(len(x) != len(y) for x, y in zip(qa, qb)):
                return False
    for pa, pb in zip(a.ports, b.ports):
        if (pa.width, pa.length, pa.layer, str(pa.direction), pa.gap_from_end) != \
                (pb.width, pb.length, pb.layer, str(pb.direction), pb.gap_from_end):
            return False
        if (pa.ground is None) != (pb.ground is None) or (pa.ground is not None and pa.ground.key(1) != pb.ground.key(1)):
            return False
    for va, vb in zip(a.vias, b.vias):
        if (va.drill, va.pad, va.top, va.bottom) != (vb.drill, vb.pad, vb.top, vb.bottom):
            return False
    return True


def _segments(layout, layer):
    """(old, new)-paired segment lists of one layer: arrays (ns, 2, 2)."""
    segs = []
    for g, n in zip(layout.polys, layout.poly_layers):
        if n != layer:
            continue
        for part in _rings(g):
            for r in part:
                segs.append(np.stack([r, np.roll(r, -1, axis=0)], 1))
    for (x0, y0, x1, y1), n in zip(layout.rects, layout.rect_layers):
        if n == layer:
            r = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            segs.append(np.stack([r, np.roll(r, -1, axis=0)], 1))
    return np.concatenate(segs) if segs else np.zeros((0, 2, 2))


def _project(P, seg):
    """Nearest segment index, parameter t and distance for each point."""
    a, b = seg[:, 0], seg[:, 1]
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-30)
    best_d = np.full(len(P), np.inf)
    best_i = np.zeros(len(P), int)
    best_t = np.zeros(len(P))
    for c0 in range(0, len(P), 2048):
        Q = P[c0:c0 + 2048]
        t = np.clip(((Q[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(Q[:, None, :] - proj, axis=2)
        i = np.argmin(d, 1)
        k = np.arange(len(Q))
        best_d[c0:c0 + 2048], best_i[c0:c0 + 2048], best_t[c0:c0 + 2048] = d[k, i], i, t[k, i]
    return best_i, best_t, best_d


def morph_mesh(old, layout):
    """Mesh of `layout` with the connectivity of the triangular mesh `old` (see module doc), or None."""
    import shapely
    from shapely.geometry import Point, box

    from .mesh import Mesh

    raw = getattr(old, "_raw", None)
    if raw is None or old.layout is None or not _compatible(old.layout, layout):
        return None
    N0, T, tl, names, cuts, snap = raw["nodes"], raw["tris"], raw["tri_layer"], raw["names"], raw["cuts"], raw["snap"]
    a, b = old.layout, layout
    node_layer = np.full(len(N0), -1)
    node_layer[T.ravel()] = np.repeat(tl, 3)
    N1 = N0.copy()
    tol_b = 2.5 * snap
    for li, name in enumerate(names):
        idx = np.nonzero(node_layer == li)[0]
        if idx.size == 0:
            continue
        P = N0[idx]
        disp = np.full((len(idx), 2), np.nan)
        # vias: rigid
        for va, vb in zip(a.vias, b.vias):
            if name not in va.pad_layers():
                continue
            near = np.linalg.norm(P - np.asarray(va.xy), axis=1) < va.pad / 2 + tol_b
            disp[near & np.isnan(disp[:, 0])] = np.asarray(vb.xy) - np.asarray(va.xy)
        # edges: same position along the segment
        sa, sb = _segments(a, name), _segments(b, name)
        edge = np.zeros(len(idx), bool)
        if len(sa):
            i, t, d = _project(P, sa)
            edge = (d < tol_b) & np.isnan(disp[:, 0])
            pa = sa[i, 0] + t[:, None] * (sa[i, 1] - sa[i, 0])
            pb = sb[i, 0] + t[:, None] * (sb[i, 1] - sb[i, 0])
            disp[edge] = (pb - pa)[edge]
        # interior: inverse-distance-squared mean of the nearest edge nodes of the same copper piece
        rest = np.isnan(disp[:, 0])
        if rest.any():
            cu = [box(*r) for r, n in zip(a.rects, a.rect_layers) if n == name]
            u = a._poly_union(name)
            if u is not None:
                cu.append(u)
            pieces = list(getattr(shapely.union_all(cu), "geoms", [shapely.union_all(cu)])) if cu else []

            def piece_of(Q):
                out = np.full(len(Q), -1)
                for k, g in enumerate(pieces):
                    gb = g.buffer(tol_b)
                    hit = shapely.contains_xy(gb, Q[:, 0], Q[:, 1]) & (out < 0)
                    out[hit] = k
                return out

            known = ~np.isnan(disp[:, 0])
            pk = piece_of(P)
            for k in np.unique(pk[rest]):
                src = np.nonzero(known & (pk == k))[0]
                dst = np.nonzero(rest & (pk == k))[0]
                if src.size == 0:
                    return None
                tree = cKDTree(P[src])
                dd, jj = tree.query(P[dst], k=min(12, src.size))
                dd, jj = np.atleast_2d(dd).reshape(len(dst), -1), np.atleast_2d(jj).reshape(len(dst), -1)
                w = 1.0 / np.maximum(dd, 1e-12) ** 2
                disp[dst] = (w[..., None] * disp[src][jj]).sum(1) / w.sum(1)[:, None]
        if np.isnan(disp).any():
            return None
        N1[idx] = P + disp
    # checks: orientation and angles kept, free edges on the new copper boundary, area per layer
    def cross(N):
        v = N[T]
        e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
        return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    c0, c1 = cross(N0), cross(N1)
    if np.any(np.sign(c0) != np.sign(c1)) or np.any(np.abs(c1) < 1e-30):
        return None

    def min_angle(N):
        v = N[T]
        L = [np.linalg.norm(v[:, (k + 1) % 3] - v[:, (k + 2) % 3], axis=1) for k in range(3)]
        return np.min(np.abs(cross(N)) / np.maximum(np.max(L, 0) ** 2, 1e-30))   # ~ sin of the smallest angle scale

    if min_angle(N1) < 0.5 * min_angle(N0):
        return None
    for li, name in enumerate(names):
        sel = tl == li
        if not sel.any():
            continue
        cu = [box(*r) for r, n in zip(b.rects, b.rect_layers) if n == name]
        u = b._poly_union(name)
        if u is not None:
            cu.append(u)
        cu += [Point(v.xy).buffer(v.pad / 2, quad_segs=16) for v in b.all_vias() if name in v.pad_layers()]
        copper = shapely.union_all(cu)
        area_mesh = 0.5 * np.abs(c1[sel]).sum()
        if abs(area_mesh - copper.area) > 0.01 * copper.area:
            return None
        Ts = T[sel]
        e = np.sort(np.concatenate([Ts[:, [0, 1]], Ts[:, [1, 2]], Ts[:, [2, 0]]]), 1)
        uq, cnt = np.unique(e, axis=0, return_counts=True)
        mid = N1[uq[cnt == 1]].mean(1)
        if len(mid) and np.max(shapely.distance(copper.boundary, shapely.points(mid))) > 5 * tol_b:
            return None
    cuts1 = dict(cuts)
    m = Mesh(N1, T.copy(), layout, None, None, tl, names, cuts1)
    m._raw = dict(raw, nodes=N1)
    return m
