"""Circuit model of the cascade: Touchstone parts, ABCD algebra, quasi-static (2-D) line table, lossy line segments.

Arrays follow the (2, 2, nf) convention for ABCD / S of 2-ports.
"""
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

C0, MU0, EPS0 = 299792458.0, 4e-7 * math.pi, 8.8541878128e-12


# ---------------------------------------------------------------- Touchstone
def read_touchstone(path):
    """(f [Hz], S (n, n, nf), z0) of a .sNp file (RI / MA / DB, any frequency unit)."""
    path = Path(path)
    n = int("".join(ch for ch in path.suffix if ch.isdigit()) or 2)
    unit, fmt, z0, rows = 1e9, "MA", 50.0, []
    for line in path.read_text(errors="replace").splitlines():
        line = line.split("!")[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            t = line.upper().split()
            unit = {"HZ": 1, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}.get(next((x for x in t if x.endswith("HZ")), "GHZ"), 1e9)
            fmt = next((x for x in t if x in ("MA", "DB", "RI")), "MA")
            if "R" in t:
                z0 = float(t[t.index("R") + 1])
            continue
        rows.extend(float(v) for v in line.split())
    d = np.array(rows).reshape(-1, 1 + 2 * n * n)
    f = d[:, 0] * unit
    a, b = d[:, 1::2], d[:, 2::2]
    s = a + 1j * b if fmt == "RI" else (a * np.exp(1j * np.radians(b)) if fmt == "MA"
                                        else 10 ** (a / 20) * np.exp(1j * np.radians(b)))
    S = np.empty((n, n, len(f)), complex)
    for k in range(n * n):
        i, j = divmod(k, n)
        if n == 2:                       # 2-port order: S11 S21 S12 S22
            i, j = [(0, 0), (1, 0), (0, 1), (1, 1)][k]
        S[i, j] = s[:, k]
    return f, S, z0


def write_s2p(path, f, S, comment=""):
    """S (nf, 2, 2), 50 ohm, RI."""
    with open(path, "w") as fh:
        for c in comment.splitlines():
            fh.write(f"! {c}\n")
        fh.write("# Hz S RI R 50\n")
        for fr, x in zip(f, S):
            fh.write(f"{fr:.0f} " + " ".join(f"{v.real:.8f} {v.imag:.8f}" for v in (x[0, 0], x[1, 0], x[0, 1], x[1, 1])) + "\n")


def interp_s(f, S, fi):
    """Magnitude and unwrapped phase interpolation onto fi (inside the data)."""
    if not (f[0] <= fi[0] + 1 and fi[-1] <= f[-1] + 1):
        raise ValueError(f"data {f[0] / 1e9}-{f[-1] / 1e9} GHz does not cover the grid")
    n = S.shape[0]
    out = np.empty((n, n, len(fi)), complex)
    for i in range(n):
        for j in range(n):
            out[i, j] = np.interp(fi, f, np.abs(S[i, j])) * np.exp(1j * np.interp(fi, f, np.unwrap(np.angle(S[i, j]))))
    return out


def s_to_abcd(S, z0=50.0):
    s11, s12, s21, s22 = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    d = 2 * s21
    return np.array([[((1 + s11) * (1 - s22) + s12 * s21) / d, z0 * ((1 + s11) * (1 + s22) - s12 * s21) / d],
                     [((1 - s11) * (1 - s22) - s12 * s21) / (z0 * d), ((1 - s11) * (1 + s22) + s12 * s21) / d]])


def abcd_to_s(M, z0=50.0):
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    den = A + B / z0 + C * z0 + D
    return np.array([[(A + B / z0 - C * z0 - D) / den, 2 * (A * D - B * C) / den],
                     [2 / den, (-A + B / z0 - C * z0 + D) / den]])


def line_abcd(z0, gamma, length):
    gl = gamma * length
    return np.array([[np.cosh(gl), z0 * np.sinh(gl)], [np.sinh(gl) / z0, np.cosh(gl)]])


def cascade(mats):
    M = mats[0]
    for m in mats[1:]:
        M = np.einsum("ijf,jkf->ikf", M, m)
    return M


def gamma(z0, eeff, w_mm, f, er, tand, sigma=5.8e7, rough=1.0):
    """Propagation constant [1/m]: conductor (surface resistance x roughness, strip-dominated) + dielectric loss."""
    rs = np.sqrt(math.pi * f * MU0 / sigma)
    ac = rough * rs / (z0 * w_mm * 1e-3)
    ad = math.pi * f / C0 * er * (eeff - 1) / (math.sqrt(eeff) * (er - 1)) * tand
    return ac + ad + 1j * 2 * math.pi * f * math.sqrt(eeff) / C0


# ---------------------------------------------------------------- 2-D quasi-static cross-section
def qs2d(width, gap, t_l1=0.0596, h12=0.0754, h23=0.082, t_l2=0.048, er12=3.06, er23=3.06, t_sr=0.020, er_sr=3.5,
         ring=0.075, dvia=0.150, air=0.60, side=0.40, res=0.0025, coarse=0.05, sr=True):
    """Z0 and eeff of a coplanar line on the signal layer: finite-volume Laplace solve (half model, PEC bottom, top and
    side walls, via rows as PEC walls). Below the line: dielectric h12, an inner foil t_l2 voided under the line and
    its gaps (resin of the layer below, present beyond the gap edge), dielectric h23, ground. Solder mask t_sr
    conformal over the copper. Units mm. Returns dict(Z0, eeff)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    W, G, RING, DV = width, gap, ring, dvia
    H_PP, H_CORE, T_CU, T_SR = h23 + t_l2, h12, t_l2, t_sr
    z_l2b, z_core, z_l1, z_top = H_PP - T_CU, H_PP, H_PP + H_CORE, H_PP + H_CORE + t_l1
    y_s, y_g = W / 2, W / 2 + G
    y_v0 = y_g + RING; y_v1 = y_v0 + DV
    Y1, Z1 = y_v1 + side, z_top + T_SR + air

    def grid(keys, lo, hi, grow=1.2):
        keys = sorted(set(np.round([k for k in keys if lo <= k <= hi], 9)))
        out = [keys[0]]
        for a0, b in zip(keys[:-1], keys[1:]):
            seg, ramp, h = b - a0, [], res
            while 2 * (sum(ramp) + h) <= seg and h <= coarse:
                ramp.append(h); h *= grow
            mid = seg - 2 * sum(ramp)
            n = int(np.ceil(mid / min(h, coarse) - 1e-9)) if mid > 1e-12 else 0
            seq = np.array(ramp + [mid / n] * n + ramp[::-1]) if n else np.array(ramp + ramp[::-1])
            if len(seq) == 0:
                seq = np.array([seg])
            out += list(a0 + np.cumsum(seq) * seg / seq.sum())
        return np.array(out)
    y = grid([0, y_s, y_s + T_SR, y_g - T_SR, y_g, y_v0, y_v1, Y1], 0, Y1)
    z = grid([0, z_l2b, z_core, z_l1, z_l1 + T_SR, z_top, z_top + T_SR, Z1], 0, Z1)
    ny, nz = len(y), len(z)
    yc, zc = 0.5 * (y[1:] + y[:-1]), 0.5 * (z[1:] + z[:-1])
    YC, ZC = np.meshgrid(yc, zc, indexing="ij")

    def eps_cells(with_diel):
        e = np.ones_like(YC)
        if not with_diel:
            return e
        e[ZC < z_core] = er23
        e[(ZC >= z_core) & (ZC < z_l1)] = er12
        if sr:
            srm = (ZC >= z_l1) & (ZC < z_top + T_SR)
            carve = (YC > y_s + T_SR) & (YC < y_g - T_SR) & (ZC > z_l1 + T_SR)
            e[srm & ~carve] = er_sr
        return e
    Yn, Zn = np.meshgrid(y, z, indexing="ij")
    tol = 1e-9
    strip = (Yn <= y_s + tol) & (Zn >= z_l1 - tol) & (Zn <= z_top + tol)
    gnd = ((Yn >= y_g - tol) & (Zn >= z_l1 - tol) & (Zn <= z_top + tol)) \
        | ((Yn >= y_g - tol) & (Zn >= z_l2b - tol) & (Zn <= z_core + tol)) \
        | ((Yn >= y_v0 - tol) & (Zn <= z_top + tol)) | (Zn <= tol) | (Zn >= Z1 - tol) | (Yn >= Y1 - tol)
    fixed = strip | gnd
    Vfix = np.where(strip, 1.0, 0.0)

    def capacitance(e):
        dy, dz = np.diff(y), np.diff(z)
        Ky = np.zeros((ny - 1, nz)); Kz = np.zeros((ny, nz - 1))
        for j in range(nz):
            if j > 0:
                Ky[:, j] += e[:, j - 1] * dz[j - 1] / 2
            if j < nz - 1:
                Ky[:, j] += e[:, j] * dz[j] / 2
        Ky /= dy[:, None]
        for i in range(ny):
            if i > 0:
                Kz[i, :] += e[i - 1, :] * dy[i - 1] / 2
            if i < ny - 1:
                Kz[i, :] += e[i, :] * dy[i] / 2
        Kz /= dz[None, :]
        idx = np.arange(ny * nz).reshape(ny, nz)
        rows, cols, vals = [], [], []
        for K, a0, b0 in ((Ky, idx[:-1, :], idx[1:, :]), (Kz, idx[:, :-1], idx[:, 1:])):
            a1, b1, k = a0.ravel(), b0.ravel(), K.ravel()
            rows += [a1, b1, a1, b1]; cols += [a1, b1, b1, a1]; vals += [k, k, -k, -k]
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nz, ny * nz))
        f = fixed.ravel(); free = ~f
        V = Vfix.ravel().copy()
        V[free] = spla.spsolve(A[free][:, free].tocsc(), -A[free][:, f] @ V[f])
        Vg = V.reshape(ny, nz)
        Wd = np.sum(Ky * np.diff(Vg, axis=0) ** 2) + np.sum(Kz * np.diff(Vg, axis=1) ** 2)
        return 2 * EPS0 * Wd
    C = capacitance(eps_cells(True))
    Ca = capacitance(eps_cells(False))
    return dict(Z0=1 / (C0 * np.sqrt(C * Ca)), eeff=C / Ca)


def _qs_job(args):
    (w, g), kw = args
    r = qs2d(w, g, **kw)
    return (w, g), (float(r["Z0"]), float(r["eeff"]))


def line_table(pairs, params, cache=None, workers=6):
    """{(w, g): (Z0, eeff)} for every (width, gap), cached in a CSV."""
    tab = {}
    if cache and Path(cache).exists():
        for r in csv.DictReader(Path(cache).open()):
            tab[(float(r["width_mm"]), float(r["gap_mm"]))] = (float(r["Z0"]), float(r["eeff"]))
    todo = [(round(w, 3), round(g, 3)) for w, g in pairs if (round(w, 3), round(g, 3)) not in tab]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for q, v in ex.map(_qs_job, [(q, params) for q in todo]):
                tab[q] = v
        if cache:
            with Path(cache).open("w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["width_mm", "gap_mm", "Z0", "eeff"])
                for (w, g), (z, e) in sorted(tab.items()):
                    wr.writerow([f"{w:.3f}", f"{g:.3f}", f"{z:.3f}", f"{e:.4f}"])
    return tab


# ---------------------------------------------------------------- the cascade of a board
class Circuit:
    """Lines from the DXF (Board.segments + qs2d table) and parts from their S2P files, on the circuit grid."""

    def __init__(self, board):
        self.b, self.cfg = board, board.cfg
        C = self.cfg["circuit"]
        self.F = np.arange(C["f_start"] * 1e9, C["f_stop"] * 1e9 + 1, C["f_step"] * 1e9)
        self.segs = None

    def tables(self):
        if self.segs is None:
            self.segs = self.b.segments()
        pairs = sorted({(w, g) for s in self.segs.values() for _, w, g in s})
        params = dict(self.b.stack.qs2d_params(), **self.cfg["circuit"].get("qs2d", {}))
        self.tab = line_table(pairs, params, cache=self.cfg.out / "line_table.csv")
        return self.tab

    def line(self, name):
        tab = self.tables()
        er, tand = self.b.stack.substrate()
        rough = self.cfg["circuit"]["roughness"]
        mats = [line_abcd(tab[(round(w, 3), round(g, 3))][0],
                          gamma(*tab[(round(w, 3), round(g, 3))], w, self.F, er, tand, self.b.stack.sigma, rough), L * 1e-3)
                for L, w, g in self.segs[name]]
        return cascade(mats)

    def part(self, e, F=None):
        F = self.F if F is None else F
        f, S, _ = read_touchstone(self.cfg.path(e["s2p"]))
        return interp_s(f, S, F)

    def chain(self, lines=None):
        """S (2, 2, nf) of the cascade; lines: {name: ABCD (2, 2, nf)} to replace the quasi-static lines."""
        seq = []
        for e in self.cfg["chain"]:
            if "line" in e:
                seq.append(lines[e["line"]] if lines and e["line"] in lines else self.line(e["line"]))
            else:
                seq.append(s_to_abcd(self.part(e)))
        return abcd_to_s(cascade(seq))
