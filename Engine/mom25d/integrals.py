"""Triangle quadrature and closed-form 1/R potential integrals (in-plane)."""

import numpy as np

# Dunavant degree-5, 7-point rule (barycentric coords, weights sum to 1)
_a1, _b1 = 0.059715871789770, 0.470142064105115
_a2, _b2 = 0.797426985353087, 0.101286507323456
TRI7_BARY = np.array([
    [1 / 3, 1 / 3, 1 / 3],
    [_a1, _b1, _b1], [_b1, _a1, _b1], [_b1, _b1, _a1],
    [_a2, _b2, _b2], [_b2, _a2, _b2], [_b2, _b2, _a2],
])
TRI7_W = np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)

TRI3_BARY = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
TRI3_W = np.full(3, 1 / 3)


def quad_points(verts, bary):
    """verts (N,3,2) -> points (N,K,2)."""
    return np.einsum("kv,nvd->nkd", bary, verts)


def potential_integrals(verts, r, d=0.0):
    """Closed-form integrals over planar triangles.

    verts : (N,3,2) counter-clockwise triangle vertices
    r     : (K,2) observation points, projected onto the triangle plane
    d     : height of the observation points above the plane (scalar)

    Returns, with R = sqrt(|r - r'|^2 + d^2),
      I0 : (N,K)     Int_T 1/R dS'
      I1 : (N,K,2)   Int_T (r'-r)/R dS'   (in-plane part)
    (Wilton et al., IEEE T-AP 1984.)
    """
    d = abs(float(d))
    d2 = d * d
    I0 = 0.0
    I1 = 0.0
    for i in range(3):
        pm = verts[:, (i + 1) % 3][:, None, :]  # edge start (N,1,2)
        pp = verts[:, (i + 2) % 3][:, None, :]  # edge end
        L = pp - pm
        ln = np.sqrt(np.sum(L * L, -1, keepdims=True))
        lhat = L / ln
        uhat = np.stack([lhat[..., 1], -lhat[..., 0]], -1)  # outward for CCW
        dm = pm - r[None]
        dp = pp - r[None]
        lm = np.sum(dm * lhat, -1)
        lp = np.sum(dp * lhat, -1)
        t = np.sum(dm * uhat, -1)  # signed distance to edge line
        Rm = np.sqrt(dm[..., 0] ** 2 + dm[..., 1] ** 2 + d2)
        Rp = np.sqrt(dp[..., 0] ** 2 + dp[..., 1] ** 2 + d2)
        t2 = t * t
        R02 = t2 + d2
        # numerically stable R + l for l < 0: (R^2 - l^2)/(R - l) = R0^2/(R - l)
        with np.errstate(divide="ignore", invalid="ignore"):
            fp = np.where(lp >= 0, Rp + lp, R02 / (Rp - lp))
            fm = np.where(lm >= 0, Rm + lm, R02 / (Rm - lm))
            logt = np.log(fp / fm)
        degenerate = np.sqrt(R02) <= 1e-12 * ln[..., 0]
        logt = np.where(degenerate, 0.0, logt)
        I0 = I0 + t * logt
        if d:
            I0 = I0 - d * (np.arctan2(t * lp, R02 + d * Rp) - np.arctan2(t * lm, R02 + d * Rm))
        I1 = I1 + 0.5 * uhat * (R02 * logt + lp * Rp - lm * Rm)[..., None]
    return I0, I1
