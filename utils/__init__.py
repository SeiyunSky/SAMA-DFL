"""
Utils package
"""
from .data_loader import (load_mnist, load_cifar10, dirichlet_partition,
                          check_dataset, check_all_datasets, download_dataset)
from .topology import (generate_ring_topology, generate_mesh_topology,
                       check_connectivity, compute_spectral_gap)

__all__ = [
    'load_mnist', 'load_cifar10', 'dirichlet_partition',
    'check_dataset', 'check_all_datasets', 'download_dataset',
    'generate_ring_topology', 'generate_mesh_topology',
    'check_connectivity', 'compute_spectral_gap'
]
