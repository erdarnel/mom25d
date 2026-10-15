"""rfpcb: config-driven EM simulation of RF PCB cascades (2.5-D MoM with mom25d).

Workflow: ODB++ -> DXF (copper painted per layer, ports drawn) -> circuit model (quasi-static lines + part S2Ps)
-> MoM2.5 line sections (de-embedded gap ports) -> merged groups / complete model (H-matrix) -> viewers.
"""
__version__ = "0.1.0"

from .config import load            # noqa: F401
