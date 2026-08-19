"""qemsel — benchmark quantum error mitigation techniques and recommend the best one per circuit.

Public API exports.
"""

from __future__ import annotations

from qemsel.api import MitigatedExecutor, run
from qemsel.features import convert_circuit_to_graph, extract_features

__version__ = "0.1.0"

__all__ = [
    "MitigatedExecutor",
    "run",
    "extract_features",
    "convert_circuit_to_graph",
]
