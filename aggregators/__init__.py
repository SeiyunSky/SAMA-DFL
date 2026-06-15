"""
Aggregators package
"""
from .base import BaseAggregator
from .sama import SAMAAggregator
from .balance import BALANCEAggregator
from .scclip import SCCLIPAggregator
from .fedavg import FedAvgAggregator
from .krum import KrumAggregator
from .trimmed_mean import TrimmedMeanAggregator
from .coord_median import CoordMedianAggregator

__all__ = [
    'BaseAggregator',
    'SAMAAggregator',
    'BALANCEAggregator',
    'SCCLIPAggregator',
    'FedAvgAggregator',
    'KrumAggregator',
    'TrimmedMeanAggregator',
    'CoordMedianAggregator',
]
