"""Interactive 3D view (Plotly HTML) of the stepped-impedance LPF: mesh,
substrate, ground, and |J| in dB on the trace and on the ground plane.

Currents are solver output (matched excitation: unit wave into port 1, port 2
matched). Ground current: quasi-static return current J_gnd = -P_h * K, see
ground_plane_current.py. Axes are true scale (1:1:1, mm). Feed lines beyond
the delta gaps are not shown (unphysical).
"""

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")

from lpf import H as h, layout, length, substrate  # noqa: E402
from mom25d import GHz, Solver, mm  # noqa: E402
from mom25d.integrals import TRI7_BARY, TRI7_W, quad_points  # noqa: E402

x_end = length()
mesh = layout().mesh(max_cell=0.6 * mm, edge_cell=0.06 * mm)

sol = Solver(mesh, substrate())
freqs = np.array([1.5, 3.0, 6.5]) * GHz
res = sol.sweep(freqs, keep_currents=True)
S = res.renormalize(50)

# displayed region: filter plus 6 mm of each feed
X0, X1, YW = -6 * mm, x_end + 6 * mm, 4 * mm
keep = (mesh.centroid[:, 0] > X0) & (mesh.centroid[:, 0] < X1)
tris = mesh.tris[keep]
nx, ny = mesh.nodes[:, 0] / mm, mesh.nodes[:, 1] / mm
wA = np.repeat(mesh.area, 3)


def nodal(v):
    num = np.bincount(mesh.tris.ravel(), weights=np.repeat(v, 3) * wA, minlength=len(mesh.nodes))
    den = np.bincount(mesh.tris.ravel(), weights=wA, minlength=len(mesh.nodes))
    return num / np.maximum(den, 1e-300)


# RWG current at 7 quadrature points per triangle (linear inside a triangle)
b = mesh.tri_basis
lab = np.where(b >= 0, mesh.edge_len[np.maximum(b, 0)], 0.0)
cA = mesh.tri_sign * lab / (2 * mesh.area[:, None])
qp = quad_points(mesh.verts, TRI7_BARY)
src = qp.reshape(-1, 2)
qw = (TRI7_W[None, :] * mesh.area[:, None]).ravel()
rv = qp[:, :, None, :] - mesh.verts[:, None, :, :]

gx = np.arange(X0, X1 + 1e-9, 0.1 * mm)
gy = np.arange(-YW, YW + 1e-9, 0.1 * mm)
GX, GY = np.meshgrid(gx, gy)
obs = np.c_[GX.ravel(), GY.ravel()]


def ground_J(xrwg, chunk=500):
    xb = np.where(b >= 0, xrwg[np.maximum(b, 0)], 0)
    Kw = np.einsum("ti,tkid,ti->tkd", cA, rv, xb).reshape(-1, 2) * qw[:, None]
    out = np.empty((len(obs), 2), complex)
    for i in range(0, len(obs), chunk):
        d = obs[i:i + chunk, None, :] - src[None]
        out[i:i + chunk] = -(h / (2 * np.pi) / (d[..., 0] ** 2 + d[..., 1] ** 2 + h * h) ** 1.5) @ Kw
    return np.sqrt(np.sum(np.abs(out) ** 2, axis=1)).reshape(GX.shape)


DB_MIN = -40.0
zt = h / mm
cbar = dict(title=dict(text="|J| [dB rel. max]"), len=0.6, x=1.0)
common = dict(x=nx, y=ny, i=tris[:, 0], j=tris[:, 1], k=tris[:, 2], flatshading=False,
              lighting=dict(ambient=0.9, diffuse=0.2, specular=0.05), hoverinfo="skip")

fig = go.Figure()
# 0 substrate (translucent box), 1 ground plane, 2 bare copper trace
bx = [X0 / mm, X1 / mm]
by = [-YW / mm, YW / mm]
vx = [bx[0], bx[1], bx[1], bx[0]] * 2
vy = [by[0], by[0], by[1], by[1]] * 2
vz = [0] * 4 + [zt] * 4
fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, alphahull=0, color="#4a8fd4", opacity=0.12,
                        name="substrate", hoverinfo="skip", showlegend=False))
fig.add_trace(go.Mesh3d(x=vx[:4], y=vy[:4], z=[-1e-3] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                        color="#b87333", opacity=1.0, name="ground", hoverinfo="skip"))
fig.add_trace(go.Mesh3d(z=np.full(len(nx), zt), color="#d4a017", name="trace", **common))

labels = []
for fi, f in enumerate(freqs):
    xm = res.matched_currents(fi, 0)
    K = sol.current_density(xm[:, None])[:, :, 0]
    kmag = nodal(np.linalg.norm(np.abs(K), axis=1))
    gmag = ground_J(xm)
    ref = max(kmag[np.unique(tris)].max(), gmag.max())
    kdb = np.clip(20 * np.log10(np.maximum(kmag, 1e-30) / ref), DB_MIN, 0)
    gdb = np.clip(20 * np.log10(np.maximum(gmag, 1e-30) / ref), DB_MIN, 0)
    fig.add_trace(go.Mesh3d(z=np.full(len(nx), zt + 1e-3), intensity=kdb, colorscale="Inferno",
                            cmin=DB_MIN, cmax=0, colorbar=cbar, showscale=True, visible=False,
                            name=f"trace |J| {f / GHz:g} GHz", **common))
    fig.add_trace(go.Surface(x=gx / mm, y=gy / mm, z=np.full(GX.shape, 1e-3), surfacecolor=gdb,
                             colorscale="Inferno", cmin=DB_MIN, cmax=0, showscale=False,
                             visible=False, name=f"ground |J| {f / GHz:g} GHz", hoverinfo="skip",
                             lighting=dict(ambient=0.9, diffuse=0.2, specular=0.05)))
    labels.append(f"{f / GHz:g} GHz  (S21 = {20 * np.log10(abs(S[fi, 1, 0])):.1f} dB)")
    print(f"{f / GHz:g} GHz done")

fig.update_layout(
    title="Stepped-impedance LPF, true scale (mm). Wave into port 1, port 2 matched.",
    scene=dict(aspectmode="data", xaxis_title="x [mm]", yaxis_title="y [mm]",
               zaxis_title="z [mm]", camera=dict(eye=dict(x=0.3, y=-1.2, z=0.9))),
    margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="#f4f4f4", showlegend=False)

# checkbox panel: trace currents / ground currents / frequency
controls = """
var gd = document.getElementById('{plot_id}');
var labels = %s;
var panel = document.createElement('div');
panel.style.cssText = 'position:absolute;top:50px;left:12px;z-index:10;background:rgba(255,255,255,0.92);' +
  'padding:10px 14px;border:1px solid #bbb;border-radius:6px;font:14px sans-serif;line-height:1.9';
var opts = labels.map(function (l, i) { return '<option value="' + i + '">' + l + '</option>'; }).join('');
panel.innerHTML = '<label><input type="checkbox" id="cbT"> Trace currents (dB)</label><br>' +
  '<label><input type="checkbox" id="cbG"> Ground-plane currents (dB)</label><br>' +
  'Frequency: <select id="selF">' + opts + '</select>';
gd.parentNode.style.position = 'relative';
gd.parentNode.insertBefore(panel, gd);
function update() {
  var t = document.getElementById('cbT').checked, g = document.getElementById('cbG').checked;
  var f = +document.getElementById('selF').value, n = gd.data.length, vis = [];
  for (var i = 0; i < n; i++) {
    if (i === 0) vis.push(true);
    else if (i === 1) vis.push(!g);
    else if (i === 2) vis.push(!t);
    else { var fi = Math.floor((i - 3) / 2), isG = (i - 3) %% 2 === 1;
           vis.push(fi === f && (isG ? g : t)); }
  }
  Plotly.restyle(gd, {visible: vis});
}
['cbT', 'cbG', 'selF'].forEach(function (id) { document.getElementById(id).onchange = update; });
""" % json.dumps(labels)
out = os.path.join(RESULTS, "lpf_3d_view.html")
fig.write_html(out, include_plotlyjs=True, post_script=controls, full_html=True,
               default_width="100%", default_height="95vh")
print("saved", out)
