"""
Utils package
"""
from .data_loader import (load_mnist, load_cifar10, dirichlet_partition,
                          check_dataset, check_all_datasets, download_dataset,
                          load_mnist_gpu, load_cifar10_gpu, GPUTensorLoader)
from .topology import (generate_ring_topology, generate_mesh_topology,
                       check_connectivity, compute_spectral_gap)
from .logger import (make_run_log_dir, open_task_log,
                     append_task_log, finalize_task_log)

__all__ = [
    'load_mnist', 'load_cifar10', 'dirichlet_partition',
    'check_dataset', 'check_all_datasets', 'download_dataset',
    'load_mnist_gpu', 'load_cifar10_gpu', 'GPUTensorLoader',
    'generate_ring_topology', 'generate_mesh_topology',
    'check_connectivity', 'compute_spectral_gap',
    'make_run_log_dir', 'open_task_log', 'append_task_log', 'finalize_task_log',
]
