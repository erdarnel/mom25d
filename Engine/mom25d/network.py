"""Connecting S-parameter networks at their ports (after a MoM sweep).

`SweepResult.connect(model, pairs)` joins ports of the simulated layout to
ports of an S-parameter block (a Touchstone file, e.g. an amplifier or a
connector placed between two layout ports) and returns the combined
`Network` on the remaining ports. Both are taken in the same real reference
impedance (the block's, normally 50 ohm; the MoM result is renormalised).

Formula (two ports of one network joined; the usual sub-network
interconnection, e.g. Compton / scikit-rf `innerconnect`): joining ports k
and l of S gives, for the remaining ports i, j,

  S'_ij = S_ij + [ S_il S_kj (1 - S_lk) + S_il S_lj S_kk
                 + S_ik S_lj (1 - S_kl) + S_ik S_kj S_ll ] / D,
  D = (1 - S_kl)(1 - S_lk) - S_kk S_ll.
"""

import numpy as np


def innerconnect(S, k, l):
    """Join ports k and l of S (nf, n, n) -> (nf, n-2, n-2)."""
    n = S.shape[-1]
    keep = [i for i in range(n) if i not in (k, l)]
    Skl, Slk, Skk, Sll = S[:, k, l], S[:, l, k], S[:, k, k], S[:, l, l]
    D = (1 - Skl) * (1 - Slk) - Skk * Sll
    Sil, Sik = S[:, keep, l][:, :, None], S[:, keep, k][:, :, None]
    Skj, Slj = S[:, k, keep][:, None, :], S[:, l, keep][:, None, :]
    num = (Sil * Skj * (1 - Slk)[:, None, None] + Sil * Slj * Skk[:, None, None]
           + Sik * Slj * (1 - Skl)[:, None, None] + Sik * Skj * Sll[:, None, None])
    return S[:, keep][:, :, keep] + num / D[:, None, None]


def connect(SA, SB, pairs):
    """Join port a of SA to port b of SB for every (a, b) in pairs.
    Result ports: SA's remaining ports, then SB's (in order)."""
    nf, na, nb = SA.shape[0], SA.shape[-1], SB.shape[-1]
    S = np.zeros((nf, na + nb, na + nb), complex)
    S[:, :na, :na], S[:, na:, na:] = SA, SB
    idx = list(range(na + nb))  # original port index of each current position
    for a, b in pairs:
        k, l = idx.index(a), idx.index(na + b)
        S = innerconnect(S, k, l)
        idx = [p for p in idx if p not in (a, na + b)]
    return S, idx


def renormalize(S, z_old, z_new):
    """S from per-port real reference impedances z_old (nf, n) to z_new."""
    nf, n = S.shape[0], S.shape[-1]
    z_old = np.broadcast_to(np.asarray(z_old, float), (nf, n))
    z_new = np.broadcast_to(np.asarray(z_new, float), (nf, n))
    out = np.empty_like(S)
    eye = np.eye(n)
    for f in range(nf):
        so, sn = np.sqrt(z_old[f]), np.sqrt(z_new[f])
        Z = (so[:, None] * np.linalg.solve(eye - S[f], eye + S[f])) * so[None, :]  # Z = D (I-S)^-1 (I+S) D
        Zn = Z / (sn[:, None] * sn[None, :])
        out[f] = np.linalg.solve(Zn + eye, Zn - eye)
    return out


class Network:
    """S-parameters on named ports with one real reference impedance."""

    def __init__(self, freqs, S, names, z0=50.0):
        self.freqs, self.S, self.names, self.z0 = np.asarray(freqs), np.asarray(S), list(names), z0

    def db(self, i, j):
        return 20 * np.log10(np.abs(self.S[:, i, j]))

    def currents(self, fi, port):
        """Layout RWG coefficients at frequency index fi for a unit incident
        wave on `port` (name or index) of this connected network, its other
        ports matched: the S-parameter blocks set what returns into the
        layout ports they are attached to (needs sweep(keep_currents=True);
        use with Solver.current_density like SweepResult.matched_currents)."""
        res, model, pairs, idx, z0 = self._source
        q = self.names.index(port) if isinstance(port, str) else int(port)
        SL = res.renormalize(z0)[fi]
        SB = model.s(self.freqs[fi])
        nl, nb = SL.shape[0], SB.shape[0]
        drive = np.zeros(nl + nb, complex)
        drive[idx[q]] = 1.0  # idx: original port numbers (layout ports first)
        # unknowns: incident waves a = [a_layout, a_block]; at a connected pair
        # (i, k): a_layout[i] = b_block[k] and a_block[k] = b_layout[i];
        # elsewhere a = drive
        Mat = np.eye(nl + nb, dtype=complex)
        for i, k in pairs:
            Mat[i, nl:] -= SB[k]
            Mat[nl + k, :nl] -= SL[i]
            drive[i] = drive[nl + k] = 0.0
        a = np.linalg.solve(Mat, drive)
        return res.wave_currents(fi, a[:nl], z0)

    def write_touchstone(self, path):
        n = len(self.names)
        with open(path, "w") as f:
            f.write("! mom25d network, ports: " + ", ".join(self.names) + "\n")
            f.write(f"# HZ S RI R {self.z0:g}\n")
            for fi, s in zip(self.freqs, self.S):
                m = s.T if n == 2 else s  # Touchstone v1 2-port order: S11 S21 S12 S22
                f.write(f"{fi:.9e} " + " ".join(f"{v.real:.9e} {v.imag:.9e}" for v in m.ravel()) + "\n")
