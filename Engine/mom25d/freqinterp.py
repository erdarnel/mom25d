"""Frequency interpolation of the MoM impedance matrix (fast fine sweeps).

The fill dominates every frequency point (e.g. 86 of 106 s at N = 8980), yet
Z changes smoothly with frequency once two things are taken out:

- the 1/(jw) of the scalar-potential term: W = jw Z = -w^2 mu <f, gA f>
  + <div f, gV div f>/eps0 + jw Zs G is smooth (polynomial-like) in w;
- the propagation phase between basis functions m and n: W~ = W exp(+j k_e
  R_mn), R_mn the 3D distance between their centres, k_e = k0 * k_ratio.

Z is filled exactly at evenly spaced anchor frequencies; at any other
frequency W~ is Lagrange-interpolated from the `order`+1 nearest anchors
(local, piecewise) and Z = W~ exp(-j k_e R) / (jw). Only the order+1
anchors in use are kept. They are stored in complex128 unless that would not
fit in about half the available memory; complex64 (N = 8980: 0.65 GB each)
costs ~5e-5 in S at low frequency, where the charge term dominates Z.
"""

import time

import numpy as np

from .constants import C0


class FreqInterp:
    """Interpolated impedance matrices for Solver.sweep(..., interp=FreqInterp(...)).

    step     : anchor spacing [Hz] (anchors span the requested band evenly);
               None: 0.35 c / (k_ratio D), D the largest distance between
               basis functions (~2 GHz for a 37 mm layout on RO4350B)
    order    : interpolation order (3 = cubic, 4 anchors per point)
    k_ratio  : phase-extraction wavenumber k_e = k_ratio * k0; None picks
               sqrt(eps_eff) of a quasi-static estimate for the stack
    check    : fill the first frequency that lies between anchors (at least
               a quarter step from both) exactly as well, and print the
               relative interpolation error of Z
    """

    def __init__(self, step=None, order=3, k_ratio=None, check=False, block=1024, dtype=None):
        self.step, self.order, self.k_ratio, self.check, self.block = step, order, k_ratio, check, block
        self.dtype = dtype  # None: complex128 if it fits, else complex64
        self.solver = None

    # ----------------------------------------------------------- setup
    def setup(self, solver, freqs):
        self.solver = solver
        f = np.asarray(freqs, float)
        lo, hi = f.min(), f.max()
        self._W = {}
        self.n_fills = 0
        self.fill_time = 0.0
        m = solver.mesh
        # basis centres: RWG edge midpoints on their level, vias at mid-span
        zl = solver.zl[m.edge_layer]
        pos = np.c_[m.edge_mid, zl]
        if m.n_vias:
            geom = solver.geometry()
            zc = ([0.5 * (a + b) for a, b in geom.via_spans] if geom.via_spans is not None
                  else np.zeros(m.n_vias))
            pos = np.r_[pos, np.c_[[v.xy for v in geom.vias], zc]]
        for fr in getattr(m, "frills", []):  # antipad frill rings: via axis at the plane
            pos = np.r_[pos, [[*fr["xy"], solver.stack.level_z(fr["plane"])]] * fr["nr"]]
        self.pos = pos
        self.nb = m.n_basis
        if self.k_ratio is None:
            self.k_ratio = float(np.sqrt(_eps_eff(solver)))
        if self.step is None:
            ext = np.ptp(pos, axis=0)
            self.step = 0.35 * C0 / (max(self.k_ratio, 1.0) * float(np.linalg.norm(ext)))
        n = max(self.order + 1, int(np.ceil((hi - lo) / self.step - 1e-9)) + 1)
        self.anchors = np.linspace(lo, hi, n) if hi > lo else lo + self.step * (np.arange(n) - n // 2)
        self._checked = not self.check
        N = len(pos)
        if self.dtype is None:
            need = (self.order + 1) * N * N * 16 + 3 * N * N * 16  # anchors + Z + LU work
            self.dtype = np.complex128 if need < 0.5 * _available_memory() else np.complex64
        return self

    def _R_block(self, r0, r1):
        d = self.pos[r0:r1, None, :] - self.pos[None, :, :]
        # float64: with float32 R the phase factor comes out complex64 and the
        # round trip exp(+jkR) exp(-jkR) loses 1e-8 -- at 1 GHz the low-
        # frequency conditioning (~1e5) turns that into 3e-3 in the currents
        R = np.sqrt((d * d).sum(-1))
        # via-via pairs: no phase extraction. Their block holds the large
        # Im gV(0) correction, which is the same for every pair (no
        # propagation phase); exp(+jkR) on it made the via-via block
        # interpolate to only 0.6-6 % (vs 1e-8 for the rest of Z)
        if r1 > self.nb:
            R[max(r0, self.nb) - r0:, self.nb:] = 0.0
        return R

    def _anchor(self, i):
        if i not in self._W:
            fa = self.anchors[i]
            t = time.time()
            Z = self.solver.impedance(fa)
            self.fill_time += time.time() - t
            self.n_fills += 1
            w = 2 * np.pi * fa
            ke = self.k_ratio * w / C0
            W = np.empty(Z.shape, self.dtype)
            for r0 in range(0, Z.shape[0], self.block):
                r1 = min(r0 + self.block, Z.shape[0])
                W[r0:r1] = 1j * w * Z[r0:r1] * np.exp(1j * ke * self._R_block(r0, r1))
            self._W[i] = W
        return self._W[i]

    def _window(self, f):
        """Indices of the order+1 anchors used at f: centred on the anchor
        interval holding f. Monotone in f, so an ascending sweep never needs
        an evicted anchor again (each anchor is filled once)."""
        n, k = len(self.anchors), self.order + 1
        j = int(np.clip(np.searchsorted(self.anchors, f, side="right") - 1, 0, n - 2))
        i0 = min(max(j - (k - 1) // 2, 0), n - k)
        return list(range(i0, i0 + k))

    def weights(self, f, idx):
        fa = self.anchors[idx]
        L = np.ones(len(idx))
        for a in range(len(idx)):
            for b in range(len(idx)):
                if a != b:
                    L[a] *= (f - fa[b]) / (fa[a] - fa[b])
        return L

    # ---------------------------------------------------------- public
    def impedance(self, f):
        """Interpolated Z at f (complex128)."""
        idx = self._window(f)
        for i in list(self._W):  # evict anchors outside the window
            if i not in idx:
                del self._W[i]
        Ws = [self._anchor(i) for i in idx]
        L = self.weights(f, idx)
        w = 2 * np.pi * f
        ke = self.k_ratio * w / C0
        N = Ws[0].shape[0]
        Z = np.empty((N, N), complex)
        for r0 in range(0, N, self.block):
            r1 = min(r0 + self.block, N)
            acc = sum(Li * Wi[r0:r1].astype(complex) for Li, Wi in zip(L, Ws))
            Z[r0:r1] = acc * np.exp(-1j * ke * self._R_block(r0, r1)) / (1j * w)
        if not self._checked and np.min(np.abs(self.anchors - f)) > 0.25 * self.step:
            self._checked = True
            self.report_error(f, Z)
        return Z

    def report_error(self, f, Zi=None):
        """Relative interpolation error of Z at f (one exact fill), per block
        (RWG, RWG-via, via-via -- the via block's large entries would hide
        the RWG block in a single norm); returns the largest."""
        Zi = self.impedance(f) if Zi is None else Zi
        Ze = self.solver.impedance(f)
        nb = self.nb
        blocks = {"RWG": (slice(0, nb), slice(0, nb)), "RWG-via": (slice(0, nb), slice(nb, None)),
                  "via-via": (slice(nb, None), slice(nb, None))}
        errs = {}
        for k, (a, b) in blocks.items():
            ne = np.linalg.norm(Ze[a, b])
            if ne > 0:
                errs[k] = np.linalg.norm(Zi[a, b] - Ze[a, b]) / ne
        err = max(errs.values())
        print(f"  FreqInterp check at {f / 1e9:.4f} GHz: relative error of Z "
              + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
              + f" (anchors every {np.diff(self.anchors).mean() / 1e9:.3g} GHz, order {self.order})")
        self.last_error = err
        return err


def _available_memory():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8e9


def _eps_eff(solver):
    """Quasi-static eps_eff of the stack under the first port's level (for
    the phase extraction only)."""
    st = solver.stack
    if solver.ports:
        layer = solver.ports[0].layer
    else:  # internal ports only: the level of the first gap port, else the first mesh layer
        gaps = {q.name: q.layer for q in solver.lumped}
        layer = next((gaps[ip["name"]] for ip in solver.iports if ip["kind"] == "gap"), solver.mesh.layer_names[0])
    e = st.eps_below(layer)
    return e if st.covered(layer) else 0.5 * (e + 1)
