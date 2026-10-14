"""Interactive 3D HTML viewers (Plotly.js from a CDN, data inlined) for any
solved model:

model_view   : the solver's geometry -- meshed copper of every layer (both
               sheets and side walls of thick metal), planes with their
               antipads, vias, ports (reference planes), lumped parts, stack
current_view : surface current density |J| of the copper, side walls, via
               barrels and planes (fields.plane_currents) for a set of cases
               (frequency x driven port), dB re the case maximum
field_view   : |E| (fields.efield) on cut planes normal to x, y or z, chosen
               with buttons and swept with a slider; 3D context + 2D map
sparams_view : S-parameter page (any number of ports): several runs and
               references (.sNp files or arrays), dB / phase, band shading,
               table view

Common options: title / subtitle / notes (HTML strings), flip (show z
mirrored, e.g. for a board solved upside down), ref (metal level shown at
z = 0), outline (shapely polygon [m] for the planes / board; default the
mesh bounding box plus 0.5 mm), focus ({name: (x, y) [m]} extra camera
buttons), port_notes / part_notes ({name: text}), stack_names (names of
the dielectric layers, bottom to top in solver order).

Cases for current_view / field_view: [dict(f=Hz, port=name, x=solution
vector, sdb=optional S matrix in dB)], e.g. from `wave_cases`.
"""

import base64
import json
import os
import subprocess

import numpy as np

from . import fields

MM = 1e-3
PALETTE = ["#c46a2f", "#b8862b", "#7a5ea8", "#2f8f8a", "#a0522d", "#5b7f2a", "#b03a6a", "#3d6fb0"]
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def wave_cases(res, fis=None, z0=50.0):
    """Cases (one per frequency index and driven port, 1 sqrt(W) incident,
    other ports matched to z0) from a SweepResult run with keep_currents."""
    fis = range(len(res.freqs)) if fis is None else fis
    S = res.renormalize(z0)
    npt = len(res.port_names)
    out = []
    for fi in fis:
        for p in range(npt):
            out.append(dict(f=float(res.freqs[fi]), port=res.port_names[p],
                            x=res.wave_currents(fi, np.eye(npt)[p], z0),
                            sdb=np.round(20 * np.log10(np.abs(S[fi]) + 1e-30), 2).tolist()))
    return out


def _tri_poly(poly, holes=(), dmax=None):
    """Triangulate a shapely polygon minus circular holes [(xy, r)] (all in
    mm) for display; dmax: largest edge (refined, for interpolated colours)."""
    import shapely
    import triangle
    from shapely.geometry import Point

    hp = [Point(xy).buffer(r, quad_segs=16) for xy, r in holes]
    g = poly.difference(shapely.union_all(hp)) if hp else poly
    g = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
    V, S, H = [], [], []
    for ring in [g.exterior, *g.interiors]:
        c = np.asarray(ring.coords)[:-1]
        n0 = len(V)
        V += c.tolist()
        S += [(n0 + i, n0 + (i + 1) % len(c)) for i in range(len(c))]
    H = [list(shapely.Polygon(r).representative_point().coords[0]) for r in g.interiors]
    d = dict(vertices=np.array(V), segments=np.array(S))
    if H:
        d["holes"] = np.array(H)
    t = triangle.triangulate(d, f"pq30a{0.5 * dmax**2:.5f}Q" if dmax else "pQ")
    return t["vertices"], t["triangles"]


def _enc(v, vmax, floor):
    """dB re vmax in [-floor, 0] -> uint8 (0 = floor, 250 = 0 dB, 255 = NaN), base64."""
    v = np.asarray(v, float)
    db = 20 * np.log10(np.maximum(np.nan_to_num(v), 1e-30) / vmax)
    q = np.clip(np.round((db + floor) / floor * 250), 0, 250).astype(np.uint8)
    q[np.isnan(v)] = 255
    return base64.b64encode(q.tobytes()).decode()


def _theme(tpl, theme):
    """Pin the page's colour scheme: "auto" follows the browser, "dark" / "light"
    force it (default $MOM25D_THEME, else auto)."""
    theme = theme or os.environ.get("MOM25D_THEME", "auto")
    if theme == "auto":
        return tpl
    if theme not in ("dark", "light"):
        raise ValueError("theme must be 'auto', 'dark' or 'light'")
    dark = theme == "dark"
    tpl = tpl.replace("@media (prefers-color-scheme: dark)", "@media all" if dark else "@media not all")
    return tpl.replace('window.matchMedia("(prefers-color-scheme: dark)").matches', "true" if dark else "false")


def _write(template, data, out, open_browser, theme=None):
    tpl = _theme(open(os.path.join(TEMPLATES, template)).read(), theme)
    with open(out, "w") as f:
        f.write(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    if open_browser:
        win = subprocess.run(["wslpath", "-w", os.path.abspath(out)], capture_output=True, text=True).stdout.strip()
        subprocess.run(["cmd.exe", "/c", "start", "", win or out], cwd="/mnt/c" if os.path.isdir("/mnt/c") else None)
    return out


class Scene:
    """Display geometry of a solved model (mm, display z)."""

    def __init__(self, solver, flip=False, ref=None, outline=None, stack_names=None, plane_dmax=0.2):
        from shapely.geometry import box

        self.sol = solver
        m, st = solver.mesh, solver.stack
        self.m, self.st = m, st
        self.sgn = -1.0 if flip else 1.0
        self.zref = st.level_z(ref) if ref else 0.0
        if outline is None:
            (x0, y0), (x1, y1) = m.nodes.min(0), m.nodes.max(0)
            outline = box(x0 - 0.5 * MM, y0 - 0.5 * MM, x1 + 0.5 * MM, y1 + 0.5 * MM)
        import shapely.affinity

        self.outline_m = outline
        self.outline = shapely.affinity.scale(outline, 1 / MM, 1 / MM, origin=(0, 0)).simplify(0.002)
        twin = getattr(m, "twin_layer", -np.ones(len(m.layer_names), int))
        self.base = {}
        for li, n in enumerate(m.layer_names):
            self.base.setdefault(n, n)
            if twin[li] >= 0:
                self.base[m.layer_names[twin[li]]] = n
        fams = list(dict.fromkeys(self.base[n] for n in m.layer_names))
        self.color = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(fams)}
        self.layers, self.node_maps = [], []
        for li, n in enumerate(m.layer_names):
            sel = np.nonzero(m.tri_layer == li)[0]
            used, inv = np.unique(m.tris[sel], return_inverse=True)
            self.layers.append(dict(name=n, fam=self.base[n], z=self.zd(st.level_z(n)), color=self.color[self.base[n]],
                                    nodes=np.round(m.nodes[used] / MM, 4).tolist(), tris=inv.reshape(-1, 3).tolist()))
            self.node_maps.append((sel, inv.reshape(-1, 3), len(used)))
        # thick metal side walls, per family
        self.wall_idx, self.walls = {}, {}
        if getattr(m, "n_walls", 0):
            for li, lj in enumerate(twin):
                if lj < 0:
                    continue
                n = m.layer_names[li]
                w = np.nonzero(m.wall & (m.edge_layer == li))[0]
                za, zb = sorted((self.zd(st.level_z(n)), self.zd(st.level_z(m.layer_names[lj]))))
                self.wall_idx[n] = w
                self.walls[n] = dict(z0=za, z1=zb, color=self.color[n],
                                     seg=np.round(m.nodes[m.edge_nodes[w]].reshape(-1, 4) / MM, 4).tolist())
        # copper outlines (boundary edges) per family, for the field view
        self.edges = {}
        for f in fams:
            li = m.layer_names.index(f)
            T = m.tris[m.tri_layer == li]
            e = np.sort(np.concatenate([T[:, [1, 2]], T[:, [2, 0]], T[:, [0, 1]]]), 1)
            u, c = np.unique(e, axis=0, return_counts=True)
            self.edges[f] = dict(z=self.zd(st.level_z(f)), color=self.color[f],
                                 seg=np.round(m.nodes[u[c == 1]].reshape(-1, 4) / MM, 4).tolist())
        # planes with their antipads
        holes = {}
        for p, xy, b in fields.antipads(m):
            holes.setdefault(p, []).append((tuple(np.asarray(xy) / MM), b / MM))
        self.plane_names = [n for n in sorted(st.planes, key=st.level_z)]
        self.planes, self.plane_nodes, self.pmeshes = [], {}, []
        cache = {}
        for n in self.plane_names:  # planes with the same holes share one display mesh
            key = tuple(holes.get(n, []))
            if key not in cache:
                V, T = _tri_poly(self.outline, key, plane_dmax)
                cache[key] = (len(self.pmeshes), V)
                self.pmeshes.append(dict(nodes=np.round(V, 4).tolist(), tris=T.tolist()))
            k, V = cache[key]
            self.plane_nodes[n] = V
            self.planes.append(dict(name=n, z=self.zd(st.level_z(n)), holes=[[list(h[0]), h[1]] for h in key], mesh=k))
        Vb, Tb = _tri_poly(self.outline)
        zd = [self.zd(z) for z, d0, d1 in self._dielectric_spans()]
        self.board = dict(nodes=np.round(Vb, 4).tolist(), tris=Tb.tolist(),
                          z0=min(zd) if zd else 0.0, z1=max(zd) if zd else 0.0)
        # vias (whole) and via segments
        self.vias = []
        for v in m.vias:
            kind = "antipad" if v.antipad is not None else ("plane" if v.connect else "via")
            za, zb = sorted((self.zd(self._z(v.top)), self.zd(self._z(v.bottom))))
            self.vias.append(dict(xy=[round(v.xy[0] / MM, 4), round(v.xy[1] / MM, 4)], r=v.drill / 2 / MM,
                                  z0=za, z1=zb, kind=kind, name=f"{v.name + ' ' if v.name else ''}{v.top}-{v.bottom}"))
        self.segs = []
        for g in m.via_segments:
            v = m.vias[g["via"]]
            za, zb = sorted((self.zd(self._z(g["top"])), self.zd(self._z(g["bottom"]))))
            self.segs.append(dict(xy=[round(v.xy[0] / MM, 4), round(v.xy[1] / MM, 4)], r=v.drill / 2 / MM,
                                  z0=za, z1=zb, name=v.name or ""))
        # ports and lumped parts
        self.ports = []
        for p in m.layout.ports:
            e = p.width / 2
            if p.ground is not None:
                e += max(gi + wi for _, gi, wi in p.ground.sides())
            self.ports.append(dict(name=p.name, layer=p.layer, z=self.zd(st.level_z(p.layer)),
                                   a=(np.asarray(p.point(0, e)) / MM).round(4).tolist(),
                                   b=(np.asarray(p.point(0, -e)) / MM).round(4).tolist(),
                                   ref=(np.asarray(p.ref) / MM).round(4).tolist(), w=round(p.width / MM, 4)))
        self.parts = []
        for q in getattr(m.layout, "lumped", []):
            val = solver.values.get(q.name, q.value) if hasattr(solver, "values") else q.value
            self.parts.append(dict(name=q.name, layer=q.layer, z=self.zd(st.level_z(q.layer)),
                                   a=(np.asarray(q.point(0)) / MM).round(4).tolist(),
                                   b=(np.asarray(q.point(q.length)) / MM).round(4).tolist(),
                                   value="open" if val is None else str(val)))
        # stack table, top (display) first
        names = stack_names or [None] * len(st.layers)
        rows = []
        lev = {k: n for n, k in st.metals.items()}
        lev.update({k: n for n, k in st.planes.items()})
        for k in range(len(st.layers) + 1):
            if k in lev or k == 0:
                n = lev.get(k, "gnd")
                rows.append(dict(kind="plane" if (n in st.planes or n == "gnd") else "cu", name=n,
                                 z=round(self.zd(float(st.z[k])), 4),
                                 color=self.color.get(self.base.get(n, ""), "")))
            if k < len(st.layers):
                d = st.layers[k]
                rows.append(dict(kind="d", name=names[k] or f"layer {k + 1}", t=round(d.t / MM, 4), er=d.eps_r,
                                 tand=d.tand))
        self.stack = rows if not flip else rows  # rows listed bottom -> top in solver z
        zs_all = [self.zd(float(z)) for z in st.z]
        self.bounds = [float(v) for v in (*self._xy_bounds(),)]
        lv = [self.zd(st.level_z(n)) for n in m.layer_names] + [p["z"] for p in self.planes]
        self.zrange = [min(lv), max(lv)] if lv else [min(zs_all), max(zs_all)]

    def zd(self, z):
        return round(self.sgn * (z - self.zref) / MM, 6)

    def _z(self, n):
        return 0.0 if n == "gnd" else self.st.level_z(n)

    def _dielectric_spans(self):
        st = self.st
        out = []
        for k, d in enumerate(st.layers):
            if d.eps_r != 1.0:
                out += [(float(st.z[k]), k, k + 1), (float(st.z[k + 1]), k, k + 1)]
        return out

    def _xy_bounds(self):
        x0, y0, x1, y1 = self.outline.bounds
        (a0, b0), (a1, b1) = self.m.nodes.min(0) / MM, self.m.nodes.max(0) / MM
        return min(x0, a0), max(x1, a1), min(y0, b0), max(y1, b1)

    def common(self, title, subtitle, notes, focus, port_notes, part_notes):
        foc = {k: [v[0] / MM, v[1] / MM, v[2] / MM if len(v) > 2 else 0.5 * sum(self.zrange)]
               for k, v in (focus or {}).items()}
        for p in self.ports:
            p["note"] = (port_notes or {}).get(p["name"], "")
        for c in self.parts:
            c["note"] = (part_notes or {}).get(c["name"], "")
        return dict(title=title, subtitle=subtitle, notes=list(notes), focus=foc, bounds=self.bounds,
                    zrange=self.zrange, layers=self.layers, walls=self.walls, planes=self.planes, pmeshes=self.pmeshes,
                    board=self.board,
                    ports=self.ports, parts=self.parts, flip=self.sgn < 0)


def model_view(solver, out, title="MoM 2.5D model", subtitle="", notes=(), flip=False, ref=None, outline=None,
               focus=None, port_notes=None, part_notes=None, stack_names=None, open_browser=False, theme=None):
    """Write the geometry viewer to `out` (HTML); returns the path."""
    sc = Scene(solver, flip, ref, outline, stack_names, plane_dmax=None)
    m = solver.mesh
    data = sc.common(title, subtitle, notes, focus, port_notes, part_notes)
    data.update(vias=sc.vias, stack=sc.stack,
                stats=dict(N=int(m.n_unknowns), nb=int(m.n_basis), nw=int(getattr(m, "n_walls", 0)),
                           nv=int(m.n_vias), nf=int(getattr(m, "n_frills", 0)), nt=int(len(m.tris)),
                           nvias=len(m.vias)))
    return _write("model_3d.html", data, out, open_browser, theme)


def current_view(solver, cases, out, title="Surface current density |J|", subtitle="", notes=(), flip=False,
                 ref=None, outline=None, focus=None, floor=80.0, plane_dx=0.1e-3, open_browser=False, theme=None,
                 verbose=True):
    """Write the current viewer for `cases` to `out`; returns the path."""
    from scipy.interpolate import RegularGridInterpolator

    sc = Scene(solver, flip, ref, outline)
    m = solver.mesh
    grid = fields.Grid(sc.bounds[0] * MM, sc.bounds[2] * MM, plane_dx, int(np.ceil((sc.bounds[1] - sc.bounds[0]) * MM / plane_dx)),
                       int(np.ceil((sc.bounds[3] - sc.bounds[2]) * MM / plane_dx)))
    nb, nv = m.n_basis, m.n_vias
    drill = np.array([m.vias[g["via"]].drill for g in m.via_segments])
    out_cases = []
    for c in cases:
        x = c["x"]
        K, _, _ = fields.current_at_quad(m, x)
        from .integrals import TRI7_W

        Ktri = (np.sqrt((np.abs(K) ** 2).sum(-1)) * TRI7_W).sum(1)
        cu = {}
        for (sel, tris, nn), L in zip(sc.node_maps, sc.layers):
            acc, wsum = np.zeros(nn), np.zeros(nn)
            for k in range(3):
                np.add.at(acc, tris[:, k], m.area[sel] * Ktri[sel])
                np.add.at(wsum, tris[:, k], m.area[sel])
            cu[L["name"]] = acc / np.maximum(wsum, 1e-30)
        Kw = {n: np.abs(x[w]) for n, w in sc.wall_idx.items()}
        Iv = np.abs(x[nb:nb + nv]) / (np.pi * drill) if nv else np.zeros(0)
        planes = {}
        if sc.plane_names:
            P = fields.plane_currents(solver, x, c["f"], grid, sc.plane_names)
            for n, v in P.items():
                a = np.sqrt((np.abs(v) ** 2).sum(0))
                planes[n] = RegularGridInterpolator((grid.yc / MM, grid.xc / MM), a, bounds_error=False,
                                                    fill_value=None)(sc.plane_nodes[n][:, ::-1])
        vals = list(cu.values()) + list(Kw.values()) + list(planes.values()) + [Iv]
        vmax = max(float(np.nanmax(v)) for v in vals if np.size(v))
        out_cases.append(dict(f=round(c["f"] / 1e9, 4), port=c["port"], vmax=vmax, sdb=c.get("sdb"),
                              cu={n: _enc(v, vmax, floor) for n, v in cu.items()},
                              walls={n: _enc(v, vmax, floor) for n, v in Kw.items()},
                              vias=_enc(Iv, vmax, floor), planes={n: _enc(v, vmax, floor) for n, v in planes.items()}))
        if verbose:
            print(f"  currents {c['f'] / 1e9:g} GHz, {c['port']}: max {vmax:.3g} A/m", flush=True)
    data = sc.common(title, subtitle, notes, focus, None, None)
    data.update(segs=sc.segs, cases=out_cases, floor=floor, ports_list=list(dict.fromkeys(c["port"] for c in cases)))
    return _write("currents_3d.html", data, out, open_browser, theme)


def field_z(solver, sc, dz=50e-6, air=0.5e-3, gap=5e-6):
    """Observation heights (solver z) for field_view: steps <= dz inside
    every layer between the lowest and highest metal level (+ air beyond),
    `gap` away from metal levels / planes, none inside thick copper."""
    st, m = solver.stack, solver.mesh
    lv = [st.level_z(n) for n in m.layer_names]
    zmin, zmax = min(lv) - air, max(lv) + air
    zmin = max(zmin, gap)
    marks = sorted(set([float(z) for z in st.z] + [zmin, zmax]))
    marks = [z for z in marks if zmin <= z <= zmax]
    thick = []
    for li, lj in enumerate(getattr(m, "twin_layer", [])):
        if lj >= 0:
            thick.append(tuple(sorted((st.level_z(m.layer_names[li]), st.level_z(m.layer_names[lj])))))
    out = []
    for a, b in zip(marks[:-1], marks[1:]):
        if any(t0 - 1e-12 <= a and b <= t1 + 1e-12 for t0, t1 in thick):
            continue
        lo, hi = a + gap, b - gap
        if hi <= lo:
            continue
        n = max(int(np.ceil((hi - lo) / dz)), 1)
        out += list(np.linspace(lo, hi, n + 1))
    return np.unique(np.round(np.array(out), 9))


def field_view(solver, cases, out, title="Electric field |E|", subtitle="", notes=(), flip=False, ref=None,
               outline=None, focus=None, floor=80.0, dx=0.1e-3, dz=50e-6, air=0.5e-3, open_browser=False, theme=None,
               verbose=True):
    """Write the cut-plane |E| viewer for `cases` to `out`; returns the path.
    0 dB = the 99.9th percentile of |E| of the case (higher values saturate)."""
    import time

    sc = Scene(solver, flip, ref, outline, plane_dmax=None)
    grid = fields.Grid(sc.bounds[0] * MM, sc.bounds[2] * MM, dx, int(np.ceil((sc.bounds[1] - sc.bounds[0]) * MM / dx)),
                       int(np.ceil((sc.bounds[3] - sc.bounds[2]) * MM / dx)))
    zs = field_z(solver, sc, dz, air, gap=dx / 4)  # charges sit on a dx / 2 grid
    zdisp = np.array([sc.zd(z) for z in zs])
    order = np.argsort(zdisp)
    out_cases = []
    greens = {}
    for c in cases:
        t0 = time.time()
        g = greens.get(c["f"]) or solver.greens(c["f"])
        greens = {c["f"]: g}
        E = fields.efield(solver, c["x"], c["f"], grid, zs, greens=g)
        A = np.sqrt((np.abs(E) ** 2).sum(0))[order]  # (nz, ny, nx), display z ascending
        vmax = float(np.percentile(A, 99.9))  # point-charge peaks next to the copper would set the scale
        out_cases.append(dict(f=round(c["f"] / 1e9, 4), port=c["port"], vmax=vmax, sdb=c.get("sdb"),
                              E=_enc(A, vmax, floor)))
        if verbose:
            print(f"  field {c['f'] / 1e9:g} GHz, {c['port']}: max |E| {vmax:.3g} V/m ({time.time() - t0:.0f} s)",
                  flush=True)
    data = sc.common(title, subtitle, notes, focus, None, None)
    data.update(cases=out_cases, floor=floor, edges=sc.edges, ports_list=list(dict.fromkeys(c["port"] for c in cases)),
                grid=dict(x=np.round(grid.xc / MM, 4).tolist(), nz=int(zs.size), y=np.round(grid.yc / MM, 4).tolist(),
                          z=np.round(zdisp[order], 5).tolist()))
    return _write("field_3d.html", data, out, open_browser, theme)


def sparams_view(series, out, ports=None, title="S-parameters", subtitle="", notes=(), band=None, key=None,
                 names=None, open_browser=False, theme=None):
    """Write an S-parameter page. series: [dict(name, path=.sNp) or dict(name,
    f=Hz, S=(nf, n, n)), optional kind="ref" for references (dashed ink)];
    ports: names (default 1..n); band: (f0, f1) [Hz] shaded; key: [(i, j)]
    shown by default (default all reflections and forward transmissions);
    names: {"i,j": label}."""
    from .lumped import Touchstone

    out_s, n = [], None
    for s in series:
        if "path" in s:
            t = Touchstone(s["path"])
            f, S = t.f, t.s_data
        else:
            f, S = np.asarray(s["f"], float), np.asarray(s["S"])
        n = S.shape[1] if n is None else n
        if S.shape[1] != n:
            raise ValueError(f"{s['name']}: {S.shape[1]} ports, expected {n}")
        out_s.append(dict(name=s["name"], kind=s.get("kind", "mom"), f=np.round(f / 1e9, 6).tolist(),
                          db=np.round(20 * np.log10(np.abs(S) + 1e-30), 3).reshape(len(f), n * n).T.tolist(),
                          ph=np.round(np.degrees(np.angle(S)), 2).reshape(len(f), n * n).T.tolist()))
    data = dict(title=title, subtitle=subtitle, notes=list(notes), series=out_s, n=n,
                ports=list(ports) if ports else [str(k + 1) for k in range(n)],
                band=[band[0] / 1e9, band[1] / 1e9] if band else None, key=[list(k) for k in key] if key else None,
                names=names or {})
    return _write("sparams.html", data, out, open_browser, theme)
