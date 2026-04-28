"""
Utils package
"""
from .data_loader import load_mnist, load_cifar10, dirichlet_partition
from .topology import (generate_ring_topology, generate_mesh_topology,
                       check_connectivity, compute_spectral_gap)

__all__ = [
    'load_mnist', 'load_cifar10', 'dirichlet_partition',
    'generate_ring_topology', 'generate_mesh_topology',
    'check_connectivity', 'compute_spectral_gap'
]
