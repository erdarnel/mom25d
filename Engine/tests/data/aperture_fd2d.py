"""2-D quasi-static FD reference: GCPW (thin L1) over L2 (thin, slot half-width a) over L3,
via walls y >= YV from L1 to L3 (as qs_planar.py). Prints Z0, eeff."""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
EPS0, C0 = 8.8541878128e-12, 299792458.0
W, G = 0.30, 0.25
H_PP, H_CORE, T1 = 0.079, 0.076, 0.002
ER_PP, ER_CORE = 3.09, 3.11
YV, Y1, AIR = 0.55, 1.6, 1.5
VIA_Z0 = 0.0
T2 = 0.0
RES, COARSE = 0.002, 0.03

def grid(keys, lo, hi, grow=1.15):
    keys = sorted(set(np.round([k for k in keys if lo <= k <= hi], 9)))
    out = [keys[0]]
    for a0, b in zip(keys[:-1], keys[1:]):
        seg, ramp, h = b - a0, [], RES
        while 2 * (sum(ramp) + h) <= seg and h <= COARSE:
            ramp.append(h); h *= grow
        mid = seg - 2 * sum(ramp)
        n = int(np.ceil(mid / min(h, COARSE) - 1e-9)) if mid > 1e-12 else 0
        seq = np.array(ramp + [mid / n] * n + ramp[::-1]) if n else np.array(ramp + ramp[::-1])
        if len(seq) == 0: seq = np.array([seg])
        out += list(a0 + np.cumsum(seq) * seg / seq.sum())
    return np.array(out)

def solve(a, ground=True):
    z_l2, z_l1 = H_PP, H_PP + H_CORE
    z_top, Z1 = z_l1 + T1, z_l1 + T1 + AIR
    ys, yg = W / 2, W / 2 + G
    ykeys = [0, ys, yg, YV, Y1] + ([a] if a not in (None, 0) else [])
    y = grid(ykeys, 0, Y1)
    z = grid([0, z_l2 - T2, z_l2, z_l1, z_top, Z1], 0, Z1)
    ny, nz = len(y), len(z)
    YC, ZC = np.meshgrid(0.5 * (y[1:] + y[:-1]), 0.5 * (z[1:] + z[:-1]), indexing="ij")
    Yn, Zn = np.meshgrid(y, z, indexing="ij")
    tol = 1e-9
    strip = (Yn <= ys + tol) & (Zn >= z_l1 - tol) & (Zn <= z_top + tol)
    gnd = (Zn <= tol) | (Zn >= Z1 - tol) | (Yn >= Y1 - tol)
    if ground:
        gnd |= ((Yn >= yg - tol) & (Zn >= z_l1 - tol) & (Zn <= z_top + tol)) | ((Yn >= YV - tol) & (Zn <= z_top + tol) & (Zn >= VIA_Z0 - tol))
    if a is not None:
        gnd |= (Yn >= a - tol) & (Zn >= z_l2 - T2 - tol) & (Zn <= z_l2 + tol)
    fixed = strip | gnd
    Vfix = np.where(strip, 1.0, 0.0)
    def eps(diel):
        e = np.ones_like(YC)
        if diel:
            e[ZC < z_l2] = ER_PP
            e[(ZC >= z_l2) & (ZC < z_l1)] = ER_CORE
        return e
    def cap(e):
        dy, dz = np.diff(y), np.diff(z)
        Ky = np.zeros((ny - 1, nz)); Kz = np.zeros((ny, nz - 1))
        Ky[:, 1:] += e * dz / 2; Ky[:, :-1] += e * dz / 2; Ky /= dy[:, None]
        Kz[1:, :] += e * dy[:, None] / 2; Kz[:-1, :] += e * dy[:, None] / 2; Kz /= dz[None, :]
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
        return 2 * EPS0 * (np.sum(Ky * np.diff(Vg, axis=0) ** 2) + np.sum(Kz * np.diff(Vg, axis=1) ** 2))
    C, Ca = cap(eps(True)), cap(eps(False))
    return 1 / (C0 * np.sqrt(C * Ca)), C / Ca

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        a = None if arg == "none" else float(arg)
        z0, ee = solve(a)
        print(f"L2 {'none' if a is None else ('solid' if a == 0 else f'slot half-width {a}')}: Z0 {z0:.2f} ohm, eeff {ee:.4f}")
