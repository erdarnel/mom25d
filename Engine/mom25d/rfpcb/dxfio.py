"""DXF input: copper read as painted, port / gap-port lines with their labels, vias, outlines, plane voids, clean-up.

Painting rule (the de-featured layout): METAL outlines combine even-odd (rings drawn inside rings are holes), the
NOT_METAL areas drawn over them are removed, and METAL islands drawn inside a NOT_METAL area (a line inside its gap
area) are added back.
"""
import math
import os
import shutil
import time
from pathlib import Path

import ezdxf
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

_CACHE = {}


def _ring(e):
    return Polygon([tuple(p[:2]) for p in e.get_points()]).buffer(0)


def even_odd(rings):
    g = Polygon()
    for r in sorted(rings, key=lambda r: -r.area):
        g = g.symmetric_difference(r)
    return g


def modelspace(dxf):
    key = ("msp", str(dxf), os.path.getmtime(dxf))
    if key not in _CACHE:
        _CACHE[key] = ezdxf.readfile(str(dxf)).modelspace()
    return _CACHE[key]


def polylines(dxf, layer):
    return [_ring(e) for e in modelspace(dxf).query(f'LWPOLYLINE[layer=="{layer}"]')]


def painted_metal(dxf, metal_layer, not_layer):
    """Effective copper of a layer by the painting rule (cached per file modification time)."""
    key = ("metal", str(dxf), os.path.getmtime(dxf), metal_layer, not_layer)
    if key in _CACHE:
        return _CACHE[key]
    metal = polylines(dxf, metal_layer)
    nots = polylines(dxf, not_layer)
    island = [m for m in metal if any(n.buffer(1e-6).contains(m) for n in nots)]
    outer = [m for m in metal if not any(m is i for i in island)]
    g = even_odd(outer).difference(even_odd(nots)).union(unary_union(island))
    _CACHE[key] = g
    return g


def voids(dxf, layer):
    """Voids of a plane layer (NOT_METAL rings, even-odd); the plane is implied everywhere else."""
    return even_odd(polylines(dxf, layer))


def outlines(dxf, layer):
    return unary_union(polylines(dxf, layer)) if polylines(dxf, layer) else Polygon()


def labelled_lines(dxf, layer):
    """LINE entities of a layer with the nearest TEXT label on the same layer: [(label, a, b)]."""
    msp = modelspace(dxf)
    lines = [(tuple(e.dxf.start)[:2], tuple(e.dxf.end)[:2]) for e in msp.query(f'LINE[layer=="{layer}"]')]
    lines = [(a, b) for a, b in lines if math.dist(a, b) > 1e-6]
    texts = []
    for t in msp.query(f'TEXT MTEXT[layer=="{layer}"]'):
        s = t.dxf.text if t.dxftype() == "TEXT" else t.text
        texts.append((s.strip(), Point(tuple(t.dxf.insert)[:2])))
    out = []
    for a, b in lines:
        ls = LineString([a, b])
        lab = min(texts, key=lambda t: ls.distance(t[1]))[0] if texts else None
        out.append((lab, a, b))
    names = [o[0] for o in out]
    dup = {n for n in names if n is not None and names.count(n) > 1}
    if dup:
        raise ValueError(f"layer {layer}: labels used by more than one line: {sorted(dup)} (move the TEXT closer)")
    return out


def gap_ports(dxf, layer, metal):
    """Drawn gap ports: a LINE from one copper piece to another. Oriented so that + is the RF line end (the piece that
    is not the biggest pour; between two smaller pieces, the smaller one): {label: dict(line=(x, y) on the line end,
    gnd=(x, y) on the ground copper, u=unit vector line -> ground, w=line width at the end, slot=length, flipped)}."""
    pieces = list(getattr(metal, "geoms", [metal]))
    pour = max(pieces, key=lambda p: p.area)
    out = {}
    for lab, a, b in labelled_lines(dxf, layer):
        on = lambda q: next((p for p in pieces if p.buffer(0.005).contains(Point(q))), None)
        pa, pb = on(a), on(b)
        if pa is None or pb is None or pa is pb:
            raise ValueError(f"{layer} line {lab} {a} -> {b}: both ends must land on two different copper pieces")
        flipped = (pa is pour) or (pb is not pour and pa.area > pb.area)
        sig, gnd, piece = (b, a, pb) if flipped else (a, b, pa)
        u = np.subtract(gnd, sig) / math.dist(sig, gnd)
        n = np.array([-u[1], u[0]])
        c = np.asarray(sig) - 0.03 * u
        ch = LineString([c - 2 * n, c + 2 * n]).intersection(piece)
        w = max((g.length for g in getattr(ch, "geoms", [ch]) if g.geom_type == "LineString"), default=0.0)
        out[lab] = dict(line=tuple(sig), gnd=tuple(gnd), u=tuple(u), w=w, slot=math.dist(sig, gnd), flipped=flipped)
    return out


def port_lines(dxf, layer):
    """PORTS lines: {label: dict(c=centre, w=length, t=unit vector along the line)}."""
    out = {}
    for lab, a, b in labelled_lines(dxf, layer):
        a, b = np.asarray(a), np.asarray(b)
        out[lab] = dict(c=tuple((a + b) / 2), w=float(np.hypot(*(b - a))), t=tuple((b - a) / np.hypot(*(b - a))))
    return out


def vias(dxf, via_layers):
    """Drill circles on the configured span layers: [(x, y, drill, pad, layer, to)]."""
    out = []
    for c in modelspace(dxf).query("CIRCLE"):
        spec = via_layers.get(c.dxf.layer)
        if spec is not None:
            out.append((c.dxf.center.x, c.dxf.center.y, 2 * c.dxf.radius, float(spec["pad"]), c.dxf.layer,
                        spec.get("to", "gnd")))
    return out


def texts(dxf, layer=None):
    msp = modelspace(dxf)
    q = "TEXT" if layer is None else f'TEXT[layer=="{layer}"]'
    return [(t.dxf.text.strip(), tuple(t.dxf.insert)[:2]) for t in msp.query(q)]


def clean(dxf, tol=1e-6, close_tol=1e-3):
    """Delete zero-length LINEs and close polylines whose ends meet (within close_tol); timestamped backup first.
    Returns (n_deleted, n_closed). Refuses if QCAD's lock file (~name) shows unsaved edits."""
    dxf = Path(dxf)
    if (dxf.parent / ("~" + dxf.name)).exists():
        raise RuntimeError(f"{dxf.name} is open with unsaved edits (QCAD ~file present)")
    doc = ezdxf.readfile(str(dxf))
    msp = doc.modelspace()
    nd = nc = 0
    for e in list(msp.query("LINE")):
        if math.dist(tuple(e.dxf.start)[:2], tuple(e.dxf.end)[:2]) < tol:
            msp.delete_entity(e)
            nd += 1
    for e in msp.query("LWPOLYLINE"):
        pts = [tuple(p[:2]) for p in e.get_points()]
        if not e.closed and len(pts) > 2 and math.dist(pts[0], pts[-1]) < close_tol:
            e.closed = True
            nc += 1
    if nd or nc:
        shutil.copy2(dxf, dxf.with_name(f"{dxf.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{dxf.suffix}"))
        doc.saveas(str(dxf))
    return nd, nc
