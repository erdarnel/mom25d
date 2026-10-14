"""mom25d -- 2.5D method of moments (MPIE + RWG) for planar multilayer circuits."""

import os as _os

# OpenBLAS oversubscribes badly on many-core/WSL machines; LU of dense MoM
# matrices is fastest with a handful of threads.  Respect user settings.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
# The variable only takes effect if numpy has not been imported yet; when
# it has (the usual `import numpy` first), OpenBLAS already runs on every
# core: a 400 x 200 complex matvec then took 2.7 ms instead of 21 us. Limit
# the running thread pool too (threadpoolctl, if installed).
try:
    from threadpoolctl import threadpool_limits as _limits

    _blas_limit = _limits(int(_os.environ["OPENBLAS_NUM_THREADS"]), user_api="blas")
except (ImportError, ValueError):
    pass

from . import analytic, lumped
from .freqinterp import FreqInterp
from .greens import GroundedSlabGreens
from .layered import GND, Dielectric, LayeredGreens, SectionedGreens, Stackup, make_greens
from .mesh import CoplanarGround, Layout, Mesh, Port, Pour, Via
from .network import Network
from .solver import (AssemblyGeometry, Solver, Substrate, SweepResult, assemble, via_impedance,
                     via_inductance)

mm = 1e-3
GHz = 1e9
