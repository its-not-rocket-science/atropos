"""Distributed utilities for multi-GPU pruning.

Provides functions to initialize distributed training environment, synchronize
metrics across processes, and manage distributed context.

Adapted from external/wanda/image_classifiers/utils.py with modifications
for pruning-specific use cases.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# Optional torch import - handle gracefully if not available
try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy objects for type checking
    torch = None  # type: ignore
    dist = None  # type: ignore


@dataclass
class DistributedConfig:
    """Configuration for distributed pruning and benchmarking.

    Attributes:
        distributed: Whether distributed mode is enabled.
        num_gpus: Number of GPUs to use.
        parallel_strategy: Parallelization strategy ("data", "layer", "model").
        distributed_backend: Distributed backend ("nccl", "gloo", "mpi").
        distributed_init_method: URL specifying how to initialize the process group.
        local_rank: Local GPU index for this process.
        rank: Global rank of this process.
        world_size: Total number of processes.
        batch_size_per_gpu: Batch size per GPU for data parallelism (default 1).
        warmup_iterations: Number of warmup iterations before benchmarking (default 5).
        benchmark_iterations: Number of iterations for throughput measurement (default 50).
        measure_scaling_efficiency: Whether to compute scaling efficiency vs
            single GPU (default True).
    """

    distributed: bool = False
    num_gpus: int = 1
    parallel_strategy: str = "data"  # "data", "layer", "model"
    distributed_backend: str = "nccl"
    distributed_init_method: str = "env://"
    local_rank: int = -1
    rank: int = -1
    world_size: int = 1
    batch_size_per_gpu: int = 1
    warmup_iterations: int = 5
    benchmark_iterations: int = 50
    measure_scaling_efficiency: bool = True


def get_rank() -> int:
    """Get current process rank.

    Returns:
        0 if distributed not initialized, otherwise rank.
    """
    if not TORCH_AVAILABLE:
        return 0
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get world size (number of processes).

    Returns:
        1 if distributed not initialized, otherwise world size.
    """
    if not TORCH_AVAILABLE:
        return 1
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if current process is main (rank 0).

    Returns:
        True if rank 0 or distributed not initialized.
    """
    return get_rank() == 0


def synchronize_metric(metric: torch.Tensor, reduce_op: str = "sum") -> torch.Tensor:
    """Synchronize a metric tensor across all processes.

    Args:
        metric: Tensor to synchronize.
        reduce_op: Reduction operation: "sum", "mean", "min", "max", "product".

    Returns:
        Synchronized tensor (same on all processes).

    Raises:
        RuntimeError: If distributed not initialized or torch not available.
    """
    if not TORCH_AVAILABLE:
        return metric
    if not dist.is_available() or not dist.is_initialized():
        return metric

    # Map reduce_op string to dist.ReduceOp
    reduce_op_map = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,  # Divide by world_size after
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "product": dist.ReduceOp.PRODUCT,
    }

    if reduce_op not in reduce_op_map:
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")

    # Ensure tensor is on GPU for NCCL
    if metric.device.type != "cuda":
        metric = metric.cuda()

    # All-reduce
    dist.all_reduce(metric, op=reduce_op_map[reduce_op])

    # For mean, divide by world size
    if reduce_op == "mean":
        metric = metric / get_world_size()

    return metric


def init_distributed_pruning(config: DistributedConfig) -> bool:
    """Initialize distributed environment for pruning.

    Args:
        config: Distributed configuration.

    Returns:
        True if distributed mode was successfully initialized.

    Raises:
        RuntimeError: If torch.distributed is not available.
    """
    if not config.distributed:
        return False

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for distributed pruning. Install with: pip install torch"
        )

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    # Determine rank and world size from environment variables
    # Support for SLURM, OMPI, and torch.distributed.launch/env vars
    if config.rank != -1:
        # Use explicit rank from config
        rank = config.rank
        world_size = config.world_size
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Standard torch.distributed.launch or torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        # OpenMPI environment
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    else:
        # Not in distributed environment
        print("Not using distributed mode (no environment variables found)")
        return False

    # Determine local rank (GPU index)
    if config.local_rank != -1:
        local_rank = config.local_rank
    elif "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        # Default: assume each process uses a different GPU
        local_rank = rank % torch.cuda.device_count()

    # Set CUDA device
    torch.cuda.set_device(local_rank)

    # Initialize process group
    print(
        f"| distributed init (rank {rank}/{world_size}): "
        f"{config.distributed_init_method}, gpu {local_rank}",
        flush=True,
    )

    dist.init_process_group(
        backend=config.distributed_backend,
        init_method=config.distributed_init_method,
        world_size=world_size,
        rank=rank,
    )

    # Synchronize all processes
    dist.barrier()

    # Update config with actual values
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank

    # Disable printing in non-main processes
    if not is_main_process():
        sys.stdout = open(os.devnull, "w")

    return True


@contextmanager
def distributed_context(config: DistributedConfig) -> Iterator[DistributedConfig]:
    """Context manager for distributed pruning operations.

    Initializes distributed environment on entry and cleans up on exit.

    Args:
        config: Distributed configuration.

    Yields:
        Updated config with rank/world_size populated.

    Example:
        with distributed_context(config) as ctx:
            # Do distributed operations
            print(f"Rank {ctx.rank} of {ctx.world_size}")
    """
    initialized = False
    try:
        initialized = init_distributed_pruning(config)
        yield config
    finally:
        if initialized and TORCH_AVAILABLE and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        # Restore stdout for non-main processes
        if initialized and not is_main_process():
            sys.stdout.close()
            sys.stdout = sys.__stdout__


def split_calibration_samples(nsamples: int, rank: int, world_size: int) -> tuple[int, int]:
    """Split calibration samples across processes for data parallelism.

    Args:
        nsamples: Total number of calibration samples.
        rank: Process rank.
        world_size: Total number of processes.

    Returns:
        Tuple of (start_index, count) for this rank's share.

    Example:
        start, count = split_calibration_samples(128, rank=0, world_size=2)
        # rank 0: (0, 64), rank 1: (64, 64)
    """
    # Simple round-robin splitting
    samples_per_rank = nsamples // world_size
    remainder = nsamples % world_size

    start = rank * samples_per_rank + min(rank, remainder)
    count = samples_per_rank + (1 if rank < remainder else 0)

    return start, count
