"""Closed-form microstrip models (zero strip thickness) used for reference."""

import numpy as np

from .constants import C0, ETA0


def eeff_static(w, h, er):
    """Hammerstad & Jensen (1980) quasi-static effective permittivity."""
    u = w / h
    a = 1 + np.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49 + np.log(1 + (u / 18.1) ** 3) / 18.7
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    return (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)


def z0_static(w, h, er):
    """Hammerstad & Jensen (1980) quasi-static characteristic impedance."""
    u = w / h
    f = 6 + (2 * np.pi - 6) * np.exp(-((30.666 / u) ** 0.7528))
    z01 = ETA0 / (2 * np.pi) * np.log(f / u + np.sqrt(1 + (2 / u) ** 2))
    return z01 / np.sqrt(eeff_static(w, h, er))


def eeff(w, h, er, f):
    """Kirschning & Jansen (1982) dispersive effective permittivity."""
    u = w / h
    fn = np.asarray(f) * h * 1e-6  # GHz*mm
    e0 = eeff_static(w, h, er)
    p1 = 0.27488 + (0.6315 + 0.525 / (1 + 0.0157 * fn) ** 20) * u - 0.065683 * np.exp(-8.7513 * u)
    p2 = 0.33622 * (1 - np.exp(-0.03442 * er))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1 - np.exp(-((fn / 38.7) ** 4.97)))
    p4 = 1 + 2.751 * (1 - np.exp(-((er / 15.916) ** 8)))
    p = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    return er - (er - e0) / (1 + p)


def beta(w, h, er, f):
    return 2 * np.pi * np.asarray(f) / C0 * np.sqrt(eeff(w, h, er, f))


def open_end_extension(w, h, er):
    """Hammerstad & Bekkadal open-end length extension."""
    u = w / h
    e = eeff_static(w, h, er)
    return 0.412 * h * (e + 0.3) * (u + 0.264) / ((e - 0.258) * (u + 0.8))


def width_for_z0(z0, h, er):
    from scipy.optimize import brentq
    return brentq(lambda w: z0_static(w, h, er) - z0, 1e-3 * h, 100 * h)
