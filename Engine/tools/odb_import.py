"""Analyse top-layer microstrip copper from an ODB++ export with mom25d.

Pick a window, select the nets to keep, give the dielectric between the
signal layer and its ground plane; every place a selected trace crosses the
window edge becomes a port (feed line outside the window, S-parameters
de-embedded to the window edge).

Examples
--------
  # which nets have copper in the window?
  python odb_import.py ODB --window X0 X1 Y0 Y1 --list-nets

  # geometry / mesh / port preview only (fast)
  python odb_import.py ODB --window X0 X1 Y0 Y1 --nets RF_NET --preview

  # full sweep
  # (dielectric thicknesses, copper weight and the top solder mask come from
  # the export; eps_r / tan d are not in ODB++ and must be given)
  python odb_import.py ODB --window X0 X1 Y0 Y1 --nets RF_NET \\
      --er 3.66 --tand 0.0037 --fmin 1 --fmax 20 --nf 20

  # two signal layers with their stackup (dielectric below each, top to bottom);
  # vias between them and to the ground come from the drill layers' spans
  python odb_import.py ODB --window ... --nets RF_NET --layers top_layer layer3 \\
      --ground-layer layer4 --dielectrics :3.66 :3.66 --preview

  # include the coplanar GND pour within 1 mm of the traces + its stitching vias
  python odb_import.py ODB --window X0 X1 Y0 Y1 --nets RF_NET \\
      --pour-nets GND --pour-band 1.0 --preview

The window can also come from select_window.py's window.json (--window-json).

Model assumptions (printed again at run time):
  * one signal layer over a solid, infinite ground plane (the next layer)
  * only the selected nets; other copper is ignored, except the pour of
    --pour-nets (e.g. GND) within --pour-band of the traces, which is tied
    to the ground plane by the board's own vias (drill layers spanning the
    signal layer and the next layer). Pour islands without vias are dropped.
  * with --pour-nets, ports with pour beside the trace at the window edge get
    GCPW feeds (measured gap / ground width, fence vias at --feed-via-pitch);
    the others, and all feeds without --pour-nets, are plain microstrip,
    so each port includes a small microstrip -> GCPW transition
  * components and signal vias are not modelled; series parts leave open gaps
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from mom25d import GHz, Solver, mm  # noqa: E402
from mom25d.odb import (ODB, build_stackup, components, find_ports, layout_from_layers,  # noqa: E402
                        place_components, port_grounds, stitched_pour)


def fill(ax, geom, **kw):
    for p in getattr(geom, "geoms", [geom]):
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        ax.fill(*p.exterior.xy, **kw)
        for r in p.interiors:
            ax.fill(*r.xy, color=ax.get_facecolor(), lw=0)


def keep_connected(copper, ports, links=()):
    """Drop copper islands that no port reaches (they'd float).

    copper : {layer: geometry}; ports : {layer: [(ref, dir, width), ...]}
    links  : [(x, y, [layers])] holes that join islands of those layers"""
    isl = [(L, p) for L, g in copper.items() for p in getattr(g, "geoms", [g]) if not p.is_empty]
    parent = list(range(len(isl)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for x, y, layers in links:
        pt = shapely.Point(x, y)
        hit = [i for i, (L, p) in enumerate(isl) if L in layers and p.distance(pt) < 1e-6]
        for i in hit[1:]:
            parent[find(i)] = find(hit[0])
    roots = {find(i) for i, (L, p) in enumerate(isl)
             for r, _, _ in ports.get(L, []) if p.distance(shapely.Point(r)) < 1e-6}
    keep = {}
    for i, (L, p) in enumerate(isl):
        if find(i) in roots:
            keep.setdefault(L, []).append(p)
    out = {L: shapely.union_all(keep[L]) if L in keep else shapely.Polygon() for L in copper}
    return out, len(isl) - sum(len(v) for v in keep.values())


def manhattan(geom, tol=1e-6):
    """True if every polygon edge is axis aligned."""
    for p in getattr(geom, "geoms", [geom]):
        for ring in [p.exterior, *p.interiors]:
            d = np.abs(np.diff(np.asarray(ring.coords), axis=0))
            if np.any((d[:, 0] > tol) & (d[:, 1] > tol)):
                return False
    return True


def port_connectivity(mesh, ports):
    """Mesh component (edge- or via-connected triangles) each port feed lands in."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nt = len(mesh.tris)
    i, j = list(mesh.tp), list(mesh.tm)
    for g in mesh.via_segments:
        if g["top_tris"] is not None and g["bottom_tris"] is not None:
            i.append(g["top_tris"][0])
            j.append(g["bottom_tris"][0])
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(nt, nt))
    _, label = connected_components(g, directed=False)
    comp = []
    for p in ports:
        xy = np.array(p.ref, float)
        xy[p.axis] = p.coord(0.5 * p.length)  # middle of the feed line
        on = np.nonzero(mesh.tri_layer == mesh.layer_names.index(p.layer))[0]
        comp.append(label[on[np.argmin(np.linalg.norm(mesh.centroid[on] - xy, axis=1))]])
    return comp


def parse_dielectrics(items):
    """T:ER[:TAND] per gap; an empty T (":3.66") takes the export's thickness."""
    out = []
    for it in items:
        v = it.split(":")
        out.append((float(v[0]) if v[0] else None, float(v[1]),
                    float(v[2]) if len(v) > 2 and v[2] else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("odb", help="extracted ODB++ directory (contains matrix/, steps/)")
    ap.add_argument("--layer", default="top_layer", help="the signal layer (single-layer model)")
    ap.add_argument("--layers", nargs="+", default=None,
                    help="several signal layers to model together (any order; sorted top to bottom)")
    ap.add_argument("--dielectrics", nargs="+", default=None, metavar="T:ER[:TAND]",
                    help="dielectric below each modelled layer, top to bottom, down to the ground "
                         "layer [mm]; empty T (':3.66') = thickness from the export")
    ap.add_argument("--window", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"), help="mm")
    ap.add_argument("--window-json", help="window.json from select_window.py")
    ap.add_argument("--nets", nargs="+", help="nets to keep (default: all copper in window)")
    ap.add_argument("--list-nets", action="store_true")
    ap.add_argument("--er", type=float, default=None,
                    help="dielectric eps_r for every gap (default: the export's DIELECTRIC layers; "
                         "Altium-style exports have none)")
    ap.add_argument("--h", type=float, default=None,
                    help="dielectric to ground plane [mm] (default: from the export)")
    ap.add_argument("--tand", type=float, default=None,
                    help="dielectric loss tangent (default: the export's, else 0)")
    ap.add_argument("--sigma", type=float, default=5.8e7, help="conductor S/m (0: PEC)")
    ap.add_argument("--t", type=float, default=None,
                    help="copper thickness [mm] (default: the export's copper weight)")
    ap.add_argument("--no-mask", action="store_true", help="ignore the top solder mask")
    ap.add_argument("--components", default=None, metavar="auto|FILE.json",
                    help="place 2-pin parts on the top layer as lumped components: values from the "
                         "export ('auto') or a JSON {refdes: 'C 1p' | 'R 10 + L 0.5n' | 'part.s2p'} "
                         "(overrides; others from the export)")
    ap.add_argument("--mask-er", type=float, default=3.5, help="solder mask eps_r (default 3.5)")
    ap.add_argument("--mask-tand", type=float, default=0.02, help="solder mask tan d (default 0.02)")
    ap.add_argument("--fmin", type=float, default=1.0, help="GHz")
    ap.add_argument("--fmax", type=float, default=20.0, help="GHz")
    ap.add_argument("--nf", type=int, default=20)
    ap.add_argument("--interp", action="store_true",
                    help="(default) fill Z only at a few anchor frequencies and interpolate")
    ap.add_argument("--no-interp", action="store_true", help="fill Z exactly at every frequency")
    ap.add_argument("--interp-step", type=float, default=None,
                    help="anchor spacing [GHz] (default: from the layout size, ~2 GHz)")
    ap.add_argument("--interp-order", type=int, default=3, help="interpolation order (default cubic)")
    ap.add_argument("--cell", type=float, default=None, help="max cell [mm] (default lambda/20 at fmax)")
    ap.add_argument("--edge-cell", type=float, default=None, help="edge refinement [mm]")
    ap.add_argument("--mesh", choices=["grid", "tri"], default="grid",
                    help="tensor grid (staircase) or unstructured triangles (exact polygons)")
    ap.add_argument("--min-angle", type=float, default=20.0, help="tri mesh: smallest interior angle [deg]")
    ap.add_argument("--max-aspect", type=float, default=5.0, help="tri mesh: edge-band cell aspect ratio")
    ap.add_argument("--feed", type=float, default=None, help="port feed length [mm]")
    ap.add_argument("--pour-nets", nargs="+", default=None,
                    help="also model the pour of these nets (e.g. GND) with its grounding vias")
    ap.add_argument("--feed-via-pitch", type=float, default=None,
                    help="GCPW feed fence-via pitch [mm] (default: median spacing of the pour vias)")
    ap.add_argument("--pour-band", type=float, default=1.0,
                    help="keep pour copper within this distance of the traces [mm]")
    ap.add_argument("--ground-layer", default=None,
                    help="the solid ground plane (default: next layer below the modelled ones)")
    ap.add_argument("--ground-nets", nargs="+", default=None,
                    help="nets whose vias connect to the ground plane (default: --pour-nets or GND)")
    ap.add_argument("--via-layers", nargs="+", default=None,
                    help="only use these drill layers (default: all plated drill layers)")
    ap.add_argument("--no-vias", action="store_true", help="ignore all vias")
    ap.add_argument("--preview", action="store_true", help="plot geometry/mesh/ports only")
    ap.add_argument("--out", default=None, help="output prefix (default ./odb_<layer>)")
    ap.add_argument("--cache", default=None, help="parsed-layer cache dir (default <odb>/../cache/mom25d)")
    a = ap.parse_args()

    if a.window_json:
        w = json.load(open(a.window_json))
        window = (*w["x"], *w["y"])
    elif a.window:
        window = tuple(a.window)
    else:
        ap.error("--window or --window-json required")
    cache = a.cache or os.path.join(os.path.dirname(os.path.abspath(a.odb.rstrip("/"))), "cache", "mom25d")
    odb = ODB(a.odb, cache_dir=cache)
    rows = odb.signal_rows()
    levels = sorted(a.layers or [a.layer], key=lambda L: rows[L])  # top to bottom
    out = a.out or f"odb_{'+'.join(levels)}"
    ground = a.ground_layer or odb.ground_layer(levels[-1])
    if rows[ground] <= rows[levels[-1]]:
        raise SystemExit(f"ground layer {ground} must be below the modelled layers {levels}")
    print(f"ODB++ step '{odb.step_name}', layers {levels} over ground {ground}, "
          f"window x {window[:2]} y {window[2:]} mm")

    if a.list_nets:
        for L in levels:
            print(f" {L}:")
            for net, area in odb.nets_in_window(L, window).items():
                print(f"  {net:30s} {area:9.3f} mm^2")
        return

    # stackup: thicknesses / copper / solder mask from the export, eps_r from the user
    given = parse_dielectrics(a.dielectrics) if a.dielectrics else None
    if given is None and a.h is not None:
        if len(levels) > 1:
            raise SystemExit("--h is for one layer; give --dielectrics for several")
        given = [(a.h, a.er, a.tand)]
    if given is not None and len(given) != len(levels):
        raise SystemExit(f"--dielectrics needs {len(levels)} entries (one below each modelled layer)")
    try:
        st, diel, report = build_stackup(odb, levels, ground, a.er, a.tand, given, not a.no_mask,
                                         a.mask_er, a.mask_tand, a.sigma or None,
                                         a.t * mm if a.t else None)
    except ValueError as e:
        raise SystemExit(f"stackup: {e}")
    print("  stackup (top to bottom):")
    for ln in report:
        print("    " + ln)
    between = [n for n, r in rows.items() if rows[levels[0]] < r < rows[ground] and n not in levels]
    if between:
        print(f"  NOTE: copper layers {between} lie inside the modelled stack and are treated as "
              "dielectric (must hold no plane under the model)")

    copper = {L: odb.copper(L, window, a.nets) for L in levels}
    ports = {L: find_ports(copper[L], window) for L in levels}
    if not any(ports.values()):
        raise SystemExit("no trace crosses the window boundary -> no ports; move the window edges across traces")
    if all(copper[L].is_empty for L in levels):
        raise SystemExit("no copper of the selected nets in the window")

    # holes: which modelled layers each one connects
    gnets = set(a.ground_nets or a.pour_nets or ["GND"])
    holes = []
    if not a.no_vias:
        vnets = (set(a.nets) | (gnets if a.pour_nets else set())) if a.nets else None
        for hv in odb.all_vias(window, vnets, a.via_layers):
            c = odb.via_connection(hv, levels, ground, gnets)
            if c is not None:
                holes.append({**hv, "top": c[0], "bottom": c[1]})
    links = [(h_["x"], h_["y"], [L for L in levels if h_["bottom"] == "gnd" or
                                  rows[h_["top"]] <= rows[L] <= rows[h_["bottom"]]])
             for h_ in holes if h_["net"] not in gnets]
    copper, dropped = keep_connected(copper, ports, links)
    if dropped:
        print(f"  dropped {dropped} copper island(s) not connected to a port")

    pours = {}
    if a.pour_nets:
        for L in levels:
            pg, _, nfloat = stitched_pour(odb, L, window, copper[L], a.pour_nets, a.pour_band, ground)
            pours[L] = pg
            print(f"  {L}: pour {'+'.join(a.pour_nets)} {pg.area:.2f} mm^2 within {a.pour_band} mm"
                  + (f", {nfloat} unstitched island(s) dropped" if nfloat else ""))

    # keep holes that land on modelled copper of their top layer (highest top wins on duplicates)
    vias, seen = [], set()
    for h_ in sorted(holes, key=lambda h_: rows[h_["top"]]):
        key = (round(h_["x"], 4), round(h_["y"], 4))
        cu = copper[h_["top"]]
        if h_["net"] in gnets and h_["top"] in pours:
            cu = pours[h_["top"]]
        if key in seen or cu.is_empty or not cu.buffer(1e-3).contains(shapely.Point(h_["x"], h_["y"])):
            continue
        seen.add(key)
        vias.append(h_)
    if vias:
        kinds = {}
        for v in vias:
            kinds[(v["top"], v["bottom"])] = kinds.get((v["top"], v["bottom"]), 0) + 1
        print("  vias: " + ", ".join(f"{n} {t}->{b}" for (t, b), n in kinds.items()))
    lost = [h_ for h_ in odb.all_vias(window, set(a.nets) if a.nets else None, a.via_layers)
            if h_["net"] not in gnets and odb.via_connection(h_, levels, ground, gnets) is None
            and any(copper[L].buffer(1e-3).contains(shapely.Point(h_["x"], h_["y"])) for L in levels)] \
        if not a.no_vias else []
    if lost:
        print(f"  WARNING: {len(lost)} via(s) on the selected nets lead to layers that are not "
              "modelled (left open)")

    for L in levels:
        modelled = copper[L] if L not in pours else copper[L].union(pours[L])
        others = odb.copper(L, window).difference(modelled.buffer(1e-4))
        gap = copper[L].distance(others) if not (others.is_empty or copper[L].is_empty) else np.inf
        hL = diel[levels.index(L)][0]
        print(f"  {L}: closest other copper (ignored): {gap:.3f} mm = {gap / hL:.1f} h")

    er_max = max(d[1] for d in diel)
    lam = 3e8 / (a.fmax * GHz) / np.sqrt(er_max)
    widths = [w for L in levels for _, _, w in ports[L]]
    hmin = min(d[0] for d in diel) * mm
    if a.cell:
        cell = a.cell * mm
    elif a.mesh == "tri" or all(manhattan(copper[L]) for L in levels if not copper[L].is_empty):
        cell = min(lam / 20, 0.5 * mm)
    else:
        # oblique edges are meshed as a staircase: cells must be small enough
        # that a diagonal trace stays edge-connected (>= ~2 cells across)
        cell = min(lam / 20, min(widths) * mm / 2.5)
        print(f"  non-Manhattan copper -> staircase mesh with {cell / mm:.3f} mm cells")
    edge = a.edge_cell * mm if a.edge_cell else 0.15 * min(widths) * mm
    htot = sum(d[0] for d in diel) * mm
    feed = a.feed * mm if a.feed else max(4 * mm, 4 * max(widths) * mm + 12 * htot, 0.4 * lam)
    # GCPW feeds: continue the pour beside each trace outside the window
    feed_vias = None
    if pours:
        pv = [v for v in vias if v["net"] in gnets]
        drill = float(np.median([v["drill"] for v in pv])) if pv else 0.3
        if a.feed_via_pitch:
            pitch = a.feed_via_pitch
        elif len(pv) > 1:
            xy = np.array([(v["x"], v["y"]) for v in pv])
            dd = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy)) * 1e9
            pitch = float(np.clip(np.median(dd.min(1)), 2 * drill, 2.0))
        else:
            pitch = 1.0
        feed_vias = dict(pitch=pitch, drill=drill)
        for L in levels:
            if L in pours:
                gr = port_grounds(pours[L], window, ports[L])
                ports[L] = [(*p_, g) for p_, g in zip(ports[L], gr)]
    lay = layout_from_layers(copper, window, feed, ports, pours, vias, feed_vias)
    if a.components:
        top = levels[0]
        if odb.signal_rows()[top] != 0:
            print(f"  NOTE: --components places top-side parts; {top} is not the top copper layer")
        else:
            table = {}
            if a.components != "auto":
                table = json.load(open(a.components))
            placed, skipped = place_components(lay, components(odb, "top"), copper[top], top, window, table)
            print(f"  components: {len(placed)} placed" + (", " + ", ".join(f"{r} {v}" for r, v in placed)
                                                          if placed else ""))
            for r, why in skipped:
                print(f"    {r} skipped: {why}")
    mesh = lay.mesh(max_cell=cell, edge_cell=edge, kind=a.mesh, min_angle=a.min_angle,
                    max_aspect=a.max_aspect)
    mesh.assign_levels(list(reversed(levels)))  # stack order, bottom to top
    qa, qr = mesh.quality()
    print(f"  {a.mesh} mesh: N = {mesh.n_unknowns}, smallest angle {qa:.1f} deg, "
          f"max aspect ratio {qr:.2f}")

    print(f"  {len(lay.ports)} port(s):")
    for p in lay.ports:
        cpw = ""
        if p.ground is not None:
            sd = {sg: (g, w) for sg, g, w in p.ground.sides()}
            cpw = ", GCPW " + " / ".join(f"{'left' if sg > 0 else 'right'} gap {sd[sg][0] / mm:.3f} "
                                         f"ground {sd[sg][1] / mm:.3f}" for sg in (1, -1) if sg in sd)
            cpw += (f" mm, fence vias {p.ground.via_pitch / mm:.2f} mm pitch"
                    if p.ground.via_pitch else " mm")
        print(f"    {p.name} on {p.layer}: ref ({p.ref[0] / mm:.3f}, {p.ref[1] / mm:.3f}) mm, dir "
              f"{p.direction}, width {p.width / mm:.3f} mm, feed {p.length / mm:.2f} mm{cpw}")
    print(f"  mesh: cell {cell / mm:.3f} mm, edge {edge / mm:.3f} mm -> {len(mesh.tris)} triangles, "
          f"{mesh.n_unknowns} unknowns ({mesh.n_vias} via segments)")
    print("  stack: " + " | ".join(f"{L} / {t:.4f} mm er {er}" for L, (t, er, _) in zip(levels, diel))
          + f" | {ground} (solid ground plane assumed under the window)")
    comp = port_connectivity(mesh, lay.ports)
    broken = len(set(comp)) > 1
    if broken:
        print(f"  WARNING: ports are in {len(set(comp))} disconnected mesh pieces {comp} -- either "
              "separate circuits, or a trace fell apart in the staircase mesh (use a smaller --cell)")

    colors = ["#d4a017", "#3fa7d6", "#e05d9b", "#7bd389"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#0b2a12")
    fill(ax, odb.copper(levels[0], window), color="#555", lw=0)
    for i, L in enumerate(levels):  # lower layers drawn translucent over the upper ones
        mesh.plot(ax, layer=L, color=colors[i % len(colors)], alpha=0.85 if i == 0 else 0.55)
    for v in lay.all_vias():
        ax.add_patch(plt.Circle((v.xy[0] / mm, v.xy[1] / mm), v.drill / 2 / mm,
                                color="red" if v.bottom == "gnd" else "white", lw=0))
    for p in lay.ports:
        x, y = p.ref[0] / mm, p.ref[1] / mm
        ax.annotate(f"{p.name} ({p.layer})", (x, y), color="cyan", fontsize=9, ha="center",
                    xytext=(0, 12), textcoords="offset points")
        ax.plot(x, y, "c^", ms=8)
    ax.add_patch(plt.Rectangle((window[0], window[2]), window[1] - window[0], window[3] - window[2],
                               fill=False, ec="r", lw=1, ls="--"))
    leg = ", ".join(f"{L}={colors[i % len(colors)]}" for i, L in enumerate(levels))
    ax.set_title(f"modelled copper ({mesh.n_unknowns} unknowns; {leg}), ignored copper of {levels[0]} grey, "
                 "red = vias to ground, white = layer-to-layer vias", fontsize=8)
    fig.tight_layout()
    fig.savefig(out + "_preview.png", dpi=130)
    print(f"  saved {out}_preview.png")
    if a.preview:
        return

    sol = Solver(mesh, st)
    f = np.linspace(a.fmin, a.fmax, a.nf) * GHz
    interp = None  # default: interpolated sweep (with an accuracy check) when it saves fills
    if not a.no_interp:
        from mom25d import FreqInterp

        interp = FreqInterp(a.interp_step * GHz if a.interp_step else None, a.interp_order,
                            check=True).setup(sol, f)
        if len(f) < 2 * len(interp.anchors):
            interp = None
    res = sol.sweep(f, interp=interp)
    n = len(lay.ports)
    res.write_touchstone(f"{out}.s{n}p", zref=50)
    S = res.renormalize(50)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n):
        for j in range(n):
            if j <= i:
                ax.plot(f / GHz, res.db(i, j, S), label=f"S{i + 1}{j + 1}", ls="-" if i != j else "--")
    ax.set(xlabel="frequency [GHz]", ylabel="dB (50 Ω)", title=f"{', '.join(a.nets or ['all nets'])}")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out + "_sparams.png", dpi=130)
    print(f"  saved {out}.s{n}p, {out}_sparams.png; port line Z0 at fmax: "
          f"{np.round(res.z0_line[-1].real, 1).tolist()} ohm")


if __name__ == "__main__":
    main()
