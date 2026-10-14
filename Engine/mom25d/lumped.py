"""Lumped components placed in a layout (`Layout.add_lumped`).

A component sits across an internal port: a short copper bridge between two
pad points with a delta gap in its middle. A load of impedance Z across the
gap adds the rank-one term Z l l^T to the MoM matrix (l: the gap edges'
signed lengths); the solver applies all loads at once with the Woodbury
identity after one solve for the external and gap excitations, so values
are cheap to change (`Solver.sweep(..., values=...)`).

One-port values: `R`, `L`, `C`, combined with `series` / `parallel`, or any
callable f -> Z [ohm]. Multi-port: `Touchstone(path)` across several gaps
(`Layout.add_component`). Port voltages are taken a -> b along each bridge.
"""

import numpy as np


class Value:
    """One-port impedance Z(f) [ohm]; `str` gives a readable description."""

    def __init__(self, fz, desc):
        self.fz, self.desc = fz, desc

    def z(self, f):
        return complex(self.fz(f))

    def __repr__(self):
        return self.desc


def _fmt(x, unit):
    for s, k in (("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9),
                 ("p", 1e-12), ("f", 1e-15)):
        if abs(x) >= k:
            return f"{x / k:g} {s}{unit}"
    return f"{x:g} {unit}"


def R(r):
    return Value(lambda f: r + 0j, _fmt(r, "ohm"))


def L(l):
    return Value(lambda f: 2j * np.pi * f * l, _fmt(l, "H"))


def C(c):
    return Value(lambda f: 1 / (2j * np.pi * f * c), _fmt(c, "F"))


def series(*vals):
    vals = [as_value(v) for v in vals]
    return Value(lambda f: sum(v.z(f) for v in vals), "(" + " + ".join(map(repr, vals)) + ")")


def parallel(*vals):
    vals = [as_value(v) for v in vals]
    return Value(lambda f: 1 / sum(1 / v.z(f) for v in vals), "(" + " || ".join(map(repr, vals)) + ")")


def as_value(v):
    """A Value from a Value, a number (ohm) or a callable f -> Z."""
    if isinstance(v, Value):
        return v
    if callable(v):
        return Value(v, getattr(v, "__name__", "Z(f)"))
    return R(float(v)) if np.isreal(v) else Value(lambda f, z=complex(v): z, f"{complex(v)} ohm")


def parse(text):
    """'R 50', 'C 1p', 'L 2.2n', 'R 10 + L 0.5n', 'C 1p || R 1k' -> Value;
    a path ending in .s2p: series_from_s2p (series-mounted part)."""
    if text.strip().lower().endswith(".s2p"):
        return series_from_s2p(text.strip())
    mult = dict(f=1e-15, p=1e-12, n=1e-9, u=1e-6, m=1e-3, k=1e3, M=1e6, G=1e9)

    def num(s):
        s = s.strip()
        if s and s[-1] in mult:
            return float(s[:-1]) * mult[s[-1]]
        return float(s)

    def one(tok):
        kind, v = tok.split()
        return {"R": R, "L": L, "C": C}[kind.upper()](num(v))
    if "||" in text:
        return parallel(*[parse(p) for p in text.split("||")])
    if "+" in text:
        return series(*[one(p) for p in text.split("+")])
    return one(text)


class Touchstone:
    """N-port component from a Touchstone file (.sNp: MA / DB / RI, any
    frequency unit). z(f) -> N x N impedance matrix, S linearly
    interpolated in frequency (real / imaginary parts)."""

    def __init__(self, path):
        self.path = path
        n = int("".join(ch for ch in path.rsplit(".", 1)[-1] if ch.isdigit()))
        fmt, unit, z0 = "MA", 1e9, 50.0
        vals = []
        for ln in open(path):
            ln = ln.split("!", 1)[0].strip()
            if not ln:
                continue
            if ln.startswith("#"):
                tok = ln[1:].upper().split()
                for i, t in enumerate(tok):
                    if t in ("HZ", "KHZ", "MHZ", "GHZ"):
                        unit = {"HZ": 1, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[t]
                    elif t in ("MA", "DB", "RI"):
                        fmt = t
                    elif t == "R":
                        z0 = float(tok[i + 1])
                continue
            vals += [float(x) for x in ln.split()]
        row = 1 + 2 * n * n
        v = np.array(vals).reshape(-1, row)
        self.f = v[:, 0] * unit
        a, b = v[:, 1::2], v[:, 2::2]
        if fmt == "RI":
            s = a + 1j * b
        elif fmt == "MA":
            s = a * np.exp(1j * np.radians(b))
        else:
            s = 10 ** (a / 20) * np.exp(1j * np.radians(b))
        s = s.reshape(-1, n, n)
        if n == 2:  # Touchstone v1 2-ports list S11 S21 S12 S22
            s = s.transpose(0, 2, 1)
        self.s_data, self.n, self.z0 = s, n, z0
        self.desc = f"Touchstone {path.rsplit('/', 1)[-1]} ({n}-port)"

    def s(self, f):
        """S at f (n x n), real / imaginary parts linearly interpolated."""
        if not self.f[0] * (1 - 1e-9) <= f <= self.f[-1] * (1 + 1e-9):
            raise ValueError(f"{self.path}: {f / 1e9:g} GHz outside {self.f[0] / 1e9:g}-{self.f[-1] / 1e9:g} GHz")
        return np.array([[np.interp(f, self.f, self.s_data[:, i, j].real)
                          + 1j * np.interp(f, self.f, self.s_data[:, i, j].imag)
                          for j in range(self.n)] for i in range(self.n)])

    def z(self, f):
        s = self.s(f)
        eye = np.eye(self.n)
        return self.z0 * (eye + s) @ np.linalg.inv(eye - s)

    def __repr__(self):
        return self.desc


def series_from_s2p(path, extrapolate=True, fit_from=None):
    """One-port Value of a series-mounted 2-terminal part measured as a
    2-port (vendor .s2p of a capacitor / inductor in series):
    Z = 2 z0 (1 - S21) / S21, interpolated within the file's band.

    extrapolate : outside the band use a series R(f) + L + C fitted to the
                  data at f >= fit_from (default f_max / 6): X = wL - 1/(wC)
                  by least squares on the relative error, R linear in f
                  (e.g. Murata GRM0225C1HR20, data to 30 GHz: C 0.2001 pF,
                  ESL 0.149 nH, SRF 29.2 GHz, |X| error < 0.1 ohm); False:
                  frequencies outside the band raise ValueError.
    The fit is in `.fit` (R0, R1 [ohm, ohm/Hz], L [H], C [F])."""
    t = Touchstone(path)
    name = path.rsplit("/", 1)[-1]

    def zdata(f):
        s = t.s(f)
        return 2 * t.z0 * (1 - s[1, 0]) / s[1, 0]
    if not extrapolate:
        return Value(zdata, f"series {name}")
    f = t.f[t.f >= (t.f[-1] / 6 if fit_from is None else fit_from)]
    z = np.array([zdata(x) for x in f])
    w = 2 * np.pi * f
    wt = 1 / np.maximum(np.abs(z.imag), 1.0)  # relative error in X
    lc = np.linalg.lstsq(np.c_[w / 1e9, -1e9 / w] * wt[:, None], z.imag * wt, rcond=None)[0]
    L, C = lc[0] * 1e-9, 1 / (lc[1] * 1e9)  # scaled columns: well conditioned
    r1, r0 = np.polyfit(f, z.real, 1)

    def fz(x):
        if t.f[0] <= x <= t.f[-1]:
            return zdata(x)
        wx = 2 * np.pi * x
        return complex(r0 + r1 * x, wx * L - 1 / (wx * C))
    v = Value(fz, f"series {name} (outside {t.f[0] / 1e9:g}-{t.f[-1] / 1e9:g} GHz: R + {L * 1e9:.3f} nH "
                  f"+ {C * 1e12:.4f} pF fit)")
    v.fit = (r0, r1, L, C)
    return v
