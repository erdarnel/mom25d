"""De-embedding of gap ports: reciprocal error boxes fitted on straight reference lines, N-port de-embedding, and
inserting a part's S-parameters between two ports.

Conventions here: S and ABCD as (..., 2, 2) / (n, n) arrays (frequency first). An error box X (ABCD) has port 1 at the
solver port and port 2 towards the DUT: measured M = X T rev(X) for a line T.
"""
import numpy as np
from scipy.optimize import least_squares


def renorm(S, z, zr=50.0):
    """S (nf, n, n) referenced to the per-port impedances z (nf, n) -> referenced to zr."""
    out = np.empty_like(S)
    I = np.eye(S.shape[1])
    for k in range(len(S)):
        r = np.diag(np.sqrt(z[k]))
        Z = r @ (I + S[k]) @ np.linalg.inv(I - S[k]) @ r
        out[k] = (Z - zr * I) @ np.linalg.inv(Z + zr * I)
    return out


def s2a(S, z=50.0):
    s11, s12, s21, s22 = S[..., 0, 0], S[..., 0, 1], S[..., 1, 0], S[..., 1, 1]
    d = 2 * s21
    A = ((1 + s11) * (1 - s22) + s12 * s21) / d
    B = z * ((1 + s11) * (1 + s22) - s12 * s21) / d
    C = ((1 - s11) * (1 - s22) - s12 * s21) / (z * d)
    D = ((1 - s11) * (1 + s22) + s12 * s21) / d
    return np.stack([np.stack([A, B], -1), np.stack([C, D], -1)], -2)


def a2s(M, z=50.0):
    A, B, C, D = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    den = A + B / z + C * z + D
    return np.stack([np.stack([(A + B / z - C * z - D) / den, 2 * (A * D - B * C) / den], -1),
                     np.stack([2 / den, (-A + B / z - C * z + D) / den], -1)], -2)


def rev(X):
    """ABCD of the reciprocal 2-port X turned round."""
    A, B, C, D = X[0, 0], X[0, 1], X[1, 0], X[1, 1]
    det = A * D - B * C
    return np.array([[D, B], [C, A]]) / det


def tline(z0, gam, L):
    """ABCD of a line: Z0 [ohm], gamma [1/m], L [mm]."""
    gl = gam * L * 1e-3
    return np.array([[np.cosh(gl), z0 * np.sinh(gl)], [np.sinh(gl) / z0, np.cosh(gl)]])


def fit_box(Ms, Ts):
    """Reciprocal X with M_i ~ X T_i rev(X) (unknowns A, B, C; D = (1 + B C) / A), least squares from X = identity.
    Returns (X, rms residual)."""
    def unpack(v):
        A, B, C = v[0] + 1j * v[1], (v[2] + 1j * v[3]) * 50, (v[4] + 1j * v[5]) / 50
        return np.array([[A, B], [C, (1 + B * C) / A]])

    def res(v):
        X = unpack(v)
        r = []
        for M, Tm in zip(Ms, Ts):
            E = (X @ Tm @ rev(X) - M) * np.array([[1, 1 / 50], [50, 1]])
            r += [E.real.ravel(), E.imag.ravel()]
        return np.concatenate(r)
    best = None
    for v0 in ([1, 0, 0, 0, 0, 0], [1, 0, 0, 0.2, 0, 0.2], [1, 0, 0, -0.2, 0, -0.2]):
        sol = least_squares(res, v0, xtol=1e-12, ftol=1e-12)
        if best is None or sol.cost < best.cost:
            best = sol
    return unpack(best.x), float(np.sqrt(best.cost))


def deembed_nport(S, Xs):
    """S (n, n, 50 ohm) measured with an error box at every port (Xs[i] ABCD, identity = none): the DUT's S.
    Attaches Q_i = a2s(inv(X_i)) with its port 2 on port i."""
    n = len(Xs)
    Q = [a2s(np.linalg.inv(X)) for X in Xs]
    Q11, Q12, Q21, Q22 = (np.diag([q[a, b] for q in Q]) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    return Q11 + Q12 @ S @ np.linalg.inv(np.eye(n) - Q22 @ S) @ Q21


def terminate(S, inner, P):
    """Connect the 2-port P between ports inner = (i, j) of S (P port 1 on i, port 2 on j); the others remain."""
    e = [k for k in range(len(S)) if k not in inner]
    inner = list(inner)
    See, Sei, Sie, Sii = S[np.ix_(e, e)], S[np.ix_(e, inner)], S[np.ix_(inner, e)], S[np.ix_(inner, inner)]
    return See + Sei @ P @ np.linalg.inv(np.eye(2) - Sii @ P) @ Sie


def box_from_refs(Ms_by_L, gam, z0):
    """Error boxes per frequency from reference lines {L mm: ABCD (nf, 2, 2)} of one port type, line gamma (nf)
    and Z0 (nf). Returns (list of X, max residual)."""
    Ls = sorted(Ms_by_L)
    Xs, rs = [], []
    for i in range(len(gam)):
        X, r = fit_box([Ms_by_L[L][i] for L in Ls], [tline(z0[i], gam[i], L) for L in Ls])
        Xs.append(X); rs.append(r)
    return Xs, max(rs)


def predict_check(Ms_by_L, gam, z0, L_check, M_check):
    """Third-length check: boxes from Ms_by_L predict the line of length L_check; max |dS| per frequency."""
    Xs, _ = box_from_refs(Ms_by_L, gam, z0)
    return np.array([np.abs(a2s(X @ tline(z0[i], gam[i], L_check) @ rev(X)) - a2s(M_check[i])).max()
                     for i, X in enumerate(Xs)])
