"""VCKO Framework - Privacy-preserving multi-centre learning."""

from .aggregator import VCKOAggregator
from .artifact import VCKOArtifact
from .builder import VCKOBuilder

__version__ = "0.1.0"

__all__ = [
    "VCKOArtifact",
    "VCKOBuilder",
    "VCKOAggregator",
]
