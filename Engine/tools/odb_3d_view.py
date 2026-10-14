"""3D view of an ODB++ board: every copper layer at its stackup height
(from the export, `ODB.stack`), the board outline as a translucent volume,
and the drilled vias.

    PYTHONPATH=.. python odb_3d_view.py ODB_DIR [--window X0 X1 Y0 Y1]
        [--layers L ...] [--simplify 0.02] [--out board_3d_view.html]

Copper outlines are simplified to --simplify mm (default 20 um) and
triangulated (shapely constrained Delaunay); vias are drawn as lines.
Writes an HTML page from odb_3d_template.html (Plotly.js from a CDN, data
inlined). Board data stays local: the page embeds the copper geometry.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import shapely
from shapely.geometry import box

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from mom25d.odb import ODB  # noqa: E402


def triangulate(geom, tol):
    """(vertices (n,2), triangles (m,3)) of polygonal geometry."""
    if geom.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 3), int)
    g = shapely.make_valid(geom.simplify(tol))
    polys = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon" and p.area > tol * tol]
    if not polys:
        return np.zeros((0, 2)), np.zeros((0, 3), int)
    tri = shapely.get_parts(shapely.constrained_delaunay_triangles(shapely.MultiPolygon(polys)))
    pts = shapely.get_coordinates(tri).reshape(-1, 4, 2)[:, :3]  # closed rings: drop the repeat
    P = np.round(pts.reshape(-1, 2), 5)
    uniq, inv = np.unique(P, axis=0, return_inverse=True)
    return uniq, inv.reshape(-1, 3)


def stack_heights(rows):
    """z [mm] of each copper layer (its top surface) and the board thickness,
    bottom of the stack at z = 0."""
    seq = []  # (name, kind, thickness) top to bottom
    for r in rows:
        if r["type"] == "copper":
            seq.append((r["name"], "copper", r["copper_mm"] or 0.035))
            if r["below_mm"]:  # vendor convention: dielectric below the copper layer
                last_cu = r is [q for q in rows if q["type"] == "copper"][-1]
                if not last_cu:
                    seq.append((r["name"] + " (below)", "dielectric", r["below_mm"]))
        elif r["thickness_mm"]:
            seq.append((r["name"], r["type"], r["thickness_mm"]))
    total = sum(t for _, _, t in seq)
    z, out, slabs = total, {}, []
    for name, kind, t in seq:
        if kind == "copper":
            out[name] = z
        slabs.append(dict(name=name, kind=kind, z0=round(z - t, 5), z1=round(z, 5)))
        z -= t
    return out, total, slabs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("odb")
    ap.add_argument("--window", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--layers", nargs="+", default=None, help="copper layers (default: all)")
    ap.add_argument("--simplify", type=float, default=0.02, help="outline tolerance [mm]")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="board_3d_view.html")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    t0 = time.time()
    odb = ODB(a.odb, cache_dir=a.cache)
    rows = odb.stack()
    zcu, total, slabs = stack_heights(rows)
    prof = odb.profile()
    win = box(a.window[0], a.window[2], a.window[1], a.window[3]) if a.window else None
    outline = prof if win is None else prof.intersection(win)
    names = [r["name"] for r in rows if r["type"] == "copper" and (a.layers is None or r["name"] in a.layers)]
    layers = []
    for n in names:
        cu = odb.copper(n, a.window)
        if win is not None:
            cu = cu.intersection(win)
        V, T = triangulate(cu, a.simplify)
        layers.append(dict(name=n, z=round(zcu[n], 5), x=V[:, 0].tolist(), y=V[:, 1].tolist(),
                           t=T.tolist(), area=round(cu.area, 2)))
        print(f"  {n:16s} z {zcu[n]:.3f} mm  {len(T):7d} triangles", flush=True)
    V, T = triangulate(outline, 0.05)
    ring = np.asarray(shapely.get_coordinates(outline.exterior if outline.geom_type == "Polygon"
                                              else max(outline.geoms, key=lambda g: g.area).exterior))
    vias = []
    for v in odb.all_vias(a.window):
        if v["start"] in zcu and v["end"] in zcu:
            vias.append([round(v["x"], 4), round(v["y"], 4), round(zcu[v["start"]], 4),
                         round(zcu[v["end"]] - (next(r["copper_mm"] for r in rows if r["name"] == v["end"]) or 0), 4),
                         int(v["start"] == names[0] and v["end"] == names[-1])])
    stack = []
    for r in rows:
        stack.append(dict(name=r["name"], kind=r["type"],
                          t=round((r["copper_mm"] if r["type"] == "copper" else r["thickness_mm"]) or 0, 4),
                          eps=r["eps"], tand=r["tand"]))
    ba = odb.board_attrs()
    x0, y0, x1, y1 = outline.bounds
    data = dict(title=a.title or os.path.basename(os.path.normpath(a.odb)), layers=layers,
                board=dict(x=V[:, 0].tolist(), y=V[:, 1].tolist(), t=T.tolist(), ring=ring.round(4).tolist(),
                           total=round(total, 4)),
                vias=vias, stack=stack, bounds=[x0, x1, y0, y1],
                info=dict(material=ba.get("_board_material", ""), standard=odb.standard_stack(),
                          n_vias=len(vias), n_tri=int(sum(len(L["t"]) for L in layers))))
    tpl = open(os.path.join(here, "odb_3d_template.html")).read()
    html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TITLE__", data["title"])
    open(a.out, "w").write(html)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB), {data['info']['n_tri']} triangles, "
          f"{len(vias)} vias, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
