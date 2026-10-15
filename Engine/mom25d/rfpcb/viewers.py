"""Self-contained dark viewers: 3-D model page (as simulated), index page, current page, cascade response page.

model_page(lay, mesh, stackup, out, title, origin, notes): true 1:1:1 scale, orthographic, sidebar toggles, "Show mesh".
"""
import html as _html
import json
from pathlib import Path

import numpy as np

MMF = 1e3          # m -> mm
LAYER_COL = ["#e08a3c", "#c8a227", "#ffb347", "#d9a066"]
PLANE_COL = {"gnd": "#8a939e"}


def _tri_region(g):
    import shapely
    xs, ys, I, J, K = [], [], [], [], []
    for p in getattr(g, "geoms", [g]):
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        for t in getattr(shapely.constrained_delaunay_triangles(p), "geoms", []):
            b = len(xs)
            for x, y in list(t.exterior.coords)[:3]:
                xs.append(x); ys.append(y)
            I.append(b); J.append(b + 1); K.append(b + 2)
    return xs, ys, I, J, K


def _cyl(items):
    """Cylinders (x, y, d, z0, z1) as one Mesh3d."""
    X, Y, Z, I, J, K = [], [], [], [], [], []
    a = np.linspace(0, 2 * np.pi, 17)[:-1]
    for x, y, d, za, zb in items:
        b = len(X)
        for zz in (za, zb):
            X += list(x + d / 2 * np.cos(a)); Y += list(y + d / 2 * np.sin(a)); Z += [zz] * 16
        for q in range(16):
            q2 = (q + 1) % 16
            I += [b + q, b + q2]; J += [b + q2, b + 16 + q2]; K += [b + 16 + q, b + 16 + q]
        for zi in (0, 16):
            for q in range(1, 15):
                I.append(b + zi); J.append(b + zi + q); K.append(b + zi + q + 1)
    return X, Y, Z, I, J, K


def model_page(lay, mesh, st, out, title, origin=(0.0, 0.0), notes=()):
    import plotly.graph_objects as go
    import plotly.offline
    from shapely import affinity
    from shapely.geometry import box
    o = np.asarray(origin, float)
    zl = lambda n: 0.0 if n == "gnd" else st.level_z(n) * MMF
    fig = go.Figure()
    meta = []                                   # (group key) per trace, for the sidebar toggles
    nodes = mesh.nodes * MMF + o
    x0, y0 = nodes.min(0) - 0.5
    x1, y1 = nodes.max(0) + 0.5
    frame = box(x0, y0, x1, y1)

    def add(tr, key):
        fig.add_trace(tr); meta.append(key)
    # copper: mesh triangles of every level, coloured per level
    for li, name in enumerate(mesh.layer_names):
        sel = np.nonzero(mesh.tri_layer == li)[0]
        if not len(sel):
            continue
        T = mesh.tris[sel]
        used, inv = np.unique(T, return_inverse=True)
        inv = inv.reshape(-1, 3)
        P = nodes[used]
        add(go.Mesh3d(x=P[:, 0], y=P[:, 1], z=np.full(len(P), zl(name)), i=inv[:, 0], j=inv[:, 1], k=inv[:, 2],
                      color=LAYER_COL[li % len(LAYER_COL)], opacity=1.0, flatshading=True, name=f"copper {name}",
                      lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), hoverinfo="name"), "copper")
        e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
        e = np.unique(e, axis=0)
        X, Y, Z = [], [], []
        for a_, b_ in e:
            X += [nodes[a_, 0], nodes[b_, 0], None]; Y += [nodes[a_, 1], nodes[b_, 1], None]
            Z += [zl(name) + 1e-4] * 2 + [None]
        add(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", line=dict(color="#111", width=1), name=f"mesh {name}",
                         visible=False, hoverinfo="skip"), "mesh")
    # planes with apertures, and the ground plane
    aps = {}
    for lo, hi, region, *_ in getattr(lay, "apertures", []):
        r = affinity.translate(affinity.scale(region, MMF, MMF, origin=(0, 0)), o[0], o[1])
        for n in {lo, hi}:
            aps[n] = r if n not in aps else aps[n].union(r)
    for n in ["gnd"] + sorted(st.planes, key=st.level_z):
        g = frame.difference(aps[n]) if n in aps else frame
        xs, ys, I, J, K = _tri_region(g)
        add(go.Mesh3d(x=xs, y=ys, z=[zl(n)] * len(xs), i=I, j=J, k=K, color=PLANE_COL.get(n, "#3f8fd2"),
                      opacity=0.45 if n == "gnd" else 0.55, flatshading=True, name=f"plane {n}" + (" (apertures open)" if n in aps else ""),
                      hoverinfo="name"), "planes")
    for gi, grp in enumerate(getattr(mesh, "ap_groups", [])):
        gm, pl = grp["mesh"], grp["planes"][0]
        V = gm.verts * MMF + o
        X, Y, Z = [], [], []
        for t in V:
            for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                X += [t[a_, 0], t[b_, 0], None]; Y += [t[a_, 1], t[b_, 1], None]; Z += [zl(pl) + 1e-4] * 2 + [None]
        add(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", line=dict(color="#7CFC00", width=1), name=f"mesh {pl} apertures",
                         visible=False, hoverinfo="skip"), "mesh")
    # vias (each via from its top to its bottom level)
    items = [(v.xy[0] * MMF + o[0], v.xy[1] * MMF + o[1], v.drill * MMF, *sorted((zl(v.top), zl(v.bottom))))
             for v in mesh.vias]
    if items:
        X, Y, Z, I, J, K = _cyl(items)
        add(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color="#d9d4c7", flatshading=True, name=f"vias ({len(items)}, true drill)",
                      hoverinfo="name"), "vias")
    # ports
    for q in getattr(lay, "lumped", []):
        a = np.asarray(q.point(0)) * MMF + o
        b = np.asarray(q.point(q.length)) * MMF + o
        z = zl(q.layer) + 2e-4
        add(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[z, z], mode="lines+text", text=["", q.name],
                         textfont=dict(color="#ff6bd6"), line=dict(color="#ff6bd6", width=10), name=f"gap port / part gap {q.name}"),
            "ports")
    for p in getattr(lay, "ports", []):
        r = np.asarray(p.ref) * MMF + o
        n = np.asarray(p.n)
        a, b = r + n * p.width * MMF / 2, r - n * p.width * MMF / 2
        z = zl(p.layer) + 2e-4
        add(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[z, z], mode="lines+text", text=["", p.name + " (feed)"],
                         textfont=dict(color="#ff6bd6"), line=dict(color="#ff6bd6", width=8), name=f"feed port {p.name}"),
            "ports")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#111418", height=860, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=False, font=dict(color="#d8dde3"),
                      scene=dict(aspectmode="data", bgcolor="#111418", camera=dict(projection=dict(type="orthographic"),
                                                                                     eye=dict(x=0.5, y=-1.0, z=1.2)),
                                 xaxis_title="x (board mm)", yaxis_title="y (board mm)", zaxis_title="z (mm)"))
    div = fig.to_html(full_html=False, include_plotlyjs=False, div_id="m3d")
    groups = [("copper", "Copper (meshed sheets)", True), ("planes", "Planes / ground", True), ("vias", "Vias", True),
              ("ports", "Ports", True), ("mesh", "Show mesh", False)]
    boxes = "".join(f'<label><input type="checkbox" id="g_{k}" {"checked" if on else ""} onchange="upd()"> {lab}</label>'
                    for k, lab, on in groups if k in meta)
    st_rows = "".join(f"<tr><td>{_html.escape(str(r))}</td></tr>" for r in notes)
    stats = (f"N = {int(mesh.n_unknowns)} (RWG {int(mesh.n_basis)}, walls {int(getattr(mesh, 'n_walls', 0))}, "
             f"via segments {int(mesh.n_vias)}, aperture {int(getattr(mesh, 'n_ap', 0))})")
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{_html.escape(title)}</title>
<script>{plotly.offline.get_plotlyjs()}</script><style>
body{{background:#111418;color:#d8dde3;font:13px system-ui,sans-serif;margin:12px}} h2{{font-size:15px;margin:0 0 8px}}
.wrap{{display:flex;gap:12px}} .side{{width:280px;flex:none}} .main{{flex:1;min-width:0}} label{{display:block;margin:5px 0}}
.side h3{{font-size:12px;color:#9aa4ae;margin:12px 0 4px;text-transform:uppercase}} td{{color:#9aa4ae;padding:2px 0}}
</style></head><body><h2>{_html.escape(title)}</h2><div class="wrap"><div class="side"><h3>Show</h3>{boxes}
<h3>Model</h3><table><tr><td>{stats}</td></tr>{st_rows}</table>
<p style="color:#9aa4ae">True 1:1:1 scale, orthographic. Drag to rotate, scroll to zoom.</p></div><div class="main">{div}</div></div>
<script>const meta={json.dumps(meta)};
function upd(){{ const g=document.getElementById('m3d');
 Plotly.restyle(g,{{visible:meta.map(k=>{{const e=document.getElementById('g_'+k); return e?e.checked:true;}})}}); }}
setTimeout(upd,200);</script></body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(page, encoding="utf-8")
    return str(out)


def index_page(items, out, title):
    """One page linking the 3-D views of every model in a batch run: items [(label, page path, note)]."""
    outp = Path(out)
    import os
    rows = "".join(f'<tr><td><a href="{_html.escape(os.path.relpath(p, outp.parent))}" target="_blank">{_html.escape(lab)}</a></td>'
                   f"<td>{_html.escape(note)}</td></tr>" for lab, p, note in items)
    outp.write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{_html.escape(title)}</title><style>
body{{background:#111418;color:#d8dde3;font:14px system-ui,sans-serif;margin:20px}} a{{color:#4fc3f7}}
td{{border-bottom:1px solid #2c323a;padding:4px 12px}} table{{border-collapse:collapse}}</style></head><body>
<h2>{_html.escape(title)}</h2><table>{rows}</table></body></html>""", encoding="utf-8")
    return str(outp)


# ---------------------------------------------------------------- currents
RANGE_DB = 50.0

def smooth_mesh(m, sel, x, o, z, db, ref, axis, name, refine=True):
    """Smooth |K| on a meshed sheet: K evaluated from the RWG coefficients at each triangle's vertices and edge
    midpoints (each triangle split into 4), the magnitudes averaged where triangles share a point, and drawn with
    vertex intensities (plotly interpolates linearly inside each small triangle)."""
    import plotly.graph_objects as go
    tri = np.nonzero(sel)[0]
    b = m.tri_basis[tri]
    cA = m.tri_sign[tri] * np.where(b >= 0, m.edge_len[np.maximum(b, 0)], 0.0) / (2 * m.area[tri][:, None])
    xb = np.where(b >= 0, x[np.maximum(b, 0)], 0)
    Vt = m.verts[tri]                                                    # (nt, 3, 2)
    bary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5]]) if refine else np.eye(3)
    pts = np.einsum("pk,tkd->tpd", bary, Vt)                             # (nt, np, 2)
    rv = pts[:, :, None, :] - Vt[:, None, :, :]                          # (nt, np, 3, 2)
    K = np.einsum("ti,tpid,ti->tpd", cA, rv, xb)
    mag = np.sqrt((np.abs(K) ** 2).sum(-1))                              # (nt, np)
    key = np.round(pts.reshape(-1, 2) * 1e9).astype(np.int64)            # nm grid: shared points coincide
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    acc = np.bincount(inv, weights=mag.ravel(), minlength=len(uk))
    cnt = np.bincount(inv, minlength=len(uk))
    val = acc / np.maximum(cnt, 1)
    idx = inv.reshape(len(tri), -1)
    if refine:                                                           # 4 sub-triangles: (0,3,5) (3,1,4) (5,4,2) (3,4,5)
        sub = np.concatenate([idx[:, [0, 3, 5]], idx[:, [3, 1, 4]], idx[:, [5, 4, 2]], idx[:, [3, 4, 5]]])
    else:
        sub = idx
    P = uk / 1e9 * 1e3 + o
    return go.Mesh3d(x=P[:, 0], y=P[:, 1], z=np.full(len(P), z), i=sub[:, 0], j=sub[:, 1], k=sub[:, 2],
                     intensity=db(val, ref), intensitymode="vertex", coloraxis=axis, name=name, flatshading=False,
                     lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), hoverinfo="skip"), float(val.max())


def via_cylinders(m, st, x, o, db, ref, zl):
    """Via segments as cylinders of their drill diameter, coloured by the barrel current density |I| / (pi d) (A/m)."""
    import plotly.graph_objects as go
    X, Y, Z, I, J, K, C = [], [], [], [], [], [], []
    ang = np.linspace(0, 2 * np.pi, 17)[:-1]
    nb = m.n_basis
    for k, g in enumerate(m.via_segments):
        v = m.vias[g["via"]]
        dens = abs(x[nb + k]) / (np.pi * v.drill)
        za, zb = sorted((zl.get(g["bottom"], 0.0), zl.get(g["top"], 0.0)))
        cx, cy = v.xy[0] * 1e3 + o[0], v.xy[1] * 1e3 + o[1]
        r = v.drill / 2 * 1e3
        b0 = len(X)
        for zz in (za, zb):
            X += list(cx + r * np.cos(ang)); Y += list(cy + r * np.sin(ang)); Z += [zz] * 16
        C += [dens] * 32
        for q in range(16):
            q2 = (q + 1) % 16
            I += [b0 + q, b0 + q2]; J += [b0 + q2, b0 + 16 + q2]; K += [b0 + 16 + q, b0 + 16 + q]
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, intensity=db(np.array(C), ref), intensitymode="vertex",
                     coloraxis="coloraxis", name="via barrels |I|/(pi d)", flatshading=True, hoverinfo="skip",
                     lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0))


def currents_page(mesh, st, cases, origin, out, title, layer):
    """Surface-current page: cases [(label, x RWG coefficients)], |K| on the sheet `layer` (smooth) and via barrels,
    dB re the largest value of each case; case buttons and layer check boxes in a left sidebar."""
    import plotly.graph_objects as go
    import plotly.offline
    o = np.asarray(origin, float)
    zl = {n: st.level_z(n) * 1e3 for n in st.metals}
    zl.update({n: st.level_z(n) * 1e3 for n in getattr(st, "planes", {})})
    zl["gnd"] = 0.0
    sel = mesh.tri_layer == mesh.layer_names.index(layer)
    db = lambda a, ref: np.clip(20 * np.log10(np.maximum(a, 1e-30) / ref), -RANGE_DB, 0)
    traces, meta = [], []
    for ci, (lab, x) in enumerate(cases):
        _, jmax = smooth_mesh(mesh, sel, x, o, 0.0, lambda a, r: a, 1.0, "coloraxis", "")
        tr, _ = smooth_mesh(mesh, sel, x, o, zl[layer], db, jmax, "coloraxis", f"{layer} |J|")
        traces.append(tr); meta.append((ci, "sheet"))
        if mesh.n_vias:
            traces.append(via_cylinders(mesh, st, x, o, db, jmax, zl)); meta.append((ci, "vias"))
    for t, (c, k) in zip(traces, meta):
        t.visible = c == 0
    fig = go.Figure(traces)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#111418", height=860, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(aspectmode="data", camera=dict(projection=dict(type="orthographic")), bgcolor="#111418",
                                 xaxis_title="x (board mm)", yaxis_title="y (board mm)", zaxis_title="z (mm)"),
                      coloraxis=dict(colorscale="Inferno", cmin=-RANGE_DB, cmax=0, colorbar=dict(title="|J| dB re max")),
                      showlegend=False, font=dict(color="#d8dde3"))
    div = fig.to_html(full_html=False, include_plotlyjs=False, div_id="cur")
    bc = "".join(f'<button onclick="setCase({i})" id="c{i}">{_html.escape(c)}</button>' for i, (c, _) in enumerate(cases))
    bl = "".join(f'<label><input type="checkbox" id="L_{k}" checked onchange="upd()"> {v}</label>'
                 for k, v in (("sheet", f"{layer} copper |J|"), ("vias", "vias |I|/(πd)")))
    html_ = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{_html.escape(title)}</title>
<script>{plotly.offline.get_plotlyjs()}</script><style>body{{background:#111418;color:#d8dde3;font:13px system-ui;margin:12px}}
.wrap{{display:flex;gap:12px}} .side{{width:280px;flex:none}} .main{{flex:1;min-width:0}} button{{background:#2c323a;color:#d8dde3;
border:1px solid #555;padding:4px 8px;margin:2px 0;display:block;width:100%;text-align:left}} button.on{{background:#6a4b12;
border-color:#c8792b}} label{{display:block;margin:4px 0}}</style></head><body><h2 style="font-size:15px">{_html.escape(title)}</h2>
<div class="wrap"><div class="side">{bc}<div style="margin-top:10px">{bl}</div></div><div class="main">{div}</div></div>
<script>const meta={json.dumps(meta)}; let cur=0; function upd(){{ const g=document.getElementById('cur');
Plotly.restyle(g,{{visible:meta.map(m=>m[0]==cur && document.getElementById('L_'+m[1]).checked)}});
document.querySelectorAll('button[id^=c]').forEach((b,i)=>b.classList.toggle('on',i==cur)); }}
function setCase(i){{cur=i;upd();}} setTimeout(upd,300);</script></body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html_, encoding="utf-8")
    return str(out)


# ---------------------------------------------------------------- cascade response
def response_page(out, title, F, curves, band=None, points=(), table_rows=(), notes=""):
    """Cascade response page: curves [(label, S (2, 2, nf), colour)] on the grid F [Hz]; points [(label, f list,
    S list (2, 2))] as markers (e.g. a complete model at a few frequencies); S21 and S11 / S22 plots, table, notes."""
    import plotly.graph_objects as go
    import plotly.offline
    db = lambda v: 20 * np.log10(np.maximum(np.abs(v), 1e-12))
    fg = np.asarray(F) / 1e9
    figs = []
    for title_, idx, yr in (("Cascade S21", [(1, 0)], None), ("Cascade S11 / S22", [(0, 0), (1, 1)], [-40, 0])):
        fi = go.Figure()
        for lab, S, col in curves:
            for (i, j), dash in zip(idx, ("solid", "dot")):
                fi.add_trace(go.Scatter(x=fg, y=db(S[i, j]), name=f"S{i + 1}{j + 1} {lab}", line=dict(color=col, dash=dash)))
        for lab, fs, Ss in points:
            for (i, j), sym in zip(idx, ("diamond", "square")):
                fi.add_trace(go.Scatter(x=np.asarray(fs) / 1e9, y=[db(s[i, j]) for s in Ss], mode="markers",
                                        name=f"S{i + 1}{j + 1} {lab}", marker=dict(size=10, symbol=sym, color="#ffffff")))
        if band:
            fi.add_vrect(x0=band[0], x1=band[1], fillcolor="#3f8fd2", opacity=0.1, line_width=0)
        fi.update_layout(template="plotly_dark", paper_bgcolor="#111418", plot_bgcolor="#111418", height=430, title=title_,
                         xaxis_title="Frequency (GHz)", yaxis=dict(title="dB", range=yr), font=dict(color="#d8dde3"),
                         legend=dict(orientation="h", y=-0.25))
        figs.append(fi.to_html(full_html=False, include_plotlyjs=False))
    rows = "".join("<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in r) + "</tr>" for r in table_rows)
    html_ = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{_html.escape(title)}</title>
<script>{plotly.offline.get_plotlyjs()}</script><style>body{{background:#111418;color:#d8dde3;font:14px system-ui;margin:20px}}
td,th{{border:1px solid #3a4048;padding:3px 8px}} table{{border-collapse:collapse}} .g{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.note{{color:#9aa4ae;max-width:1150px}}</style></head><body><h2>{_html.escape(title)}</h2><div class="g">{figs[0]}{figs[1]}</div>
<table>{rows}</table><p class="note">{_html.escape(notes)}</p></body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html_, encoding="utf-8")
    return str(out)
