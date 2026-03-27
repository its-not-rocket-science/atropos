"""Distributed benchmarking for multi-GPU model validation.

Provides DistributedBenchmarkWrapper for measuring model performance across
multiple GPUs with data parallelism, computing scaling efficiency and
per-GPU metrics.
"""

from __future__ import annotations

import time
from typing import Any

from ..distributed_utils import DistributedConfig, distributed_context, synchronize_metric
from ..logging_config import get_logger
from .models import MeasuredMetrics

logger = get_logger("distributed_benchmark")


class DistributedBenchmarkWrapper:
    """Wrapper for multi-GPU benchmarking of models.

    This class sets up distributed data parallelism and measures model
    performance across multiple GPUs. It computes scaling efficiency,
    communication overhead, and per-GPU memory usage.

    Example:
        wrapper = DistributedBenchmarkWrapper(gpu_count=2, parallel_strategy="data")
        metrics = wrapper.benchmark_model(model, tokenizer, device="cuda")
    """

    def __init__(
        self,
        gpu_count: int = 1,
        parallel_strategy: str = "data",
        batch_size_per_gpu: int = 1,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 50,
        measure_scaling_efficiency: bool = True,
        distributed_backend: str = "nccl",
        distributed_init_method: str = "env://",
    ):
        """Initialize distributed benchmark wrapper.

        Args:
            gpu_count: Number of GPUs to use for benchmarking.
            parallel_strategy: Parallelization strategy ("data", "layer", "model").
                Only "data" parallelism is currently implemented.
            batch_size_per_gpu: Batch size per GPU for data parallelism.
            warmup_iterations: Number of warmup iterations before benchmarking.
            benchmark_iterations: Number of iterations for throughput measurement.
            measure_scaling_efficiency: Whether to compute scaling efficiency
                compared to single GPU.
            distributed_backend: PyTorch distributed backend ("nccl", "gloo").
            distributed_init_method: URL for process group initialization.
        """
        self.gpu_count = gpu_count
        self.parallel_strategy = parallel_strategy
        self.batch_size_per_gpu = batch_size_per_gpu
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.measure_scaling_efficiency = measure_scaling_efficiency
        self.distributed_backend = distributed_backend
        self.distributed_init_method = distributed_init_method

        # Will be populated during benchmarking
        self.distributed_config: DistributedConfig | None = None
        self._single_gpu_throughput: float | None = None

    def benchmark_model(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        model_name: str = "unknown",
        batch_size: int = 1,
        tokens_per_request: int = 512,
    ) -> MeasuredMetrics:
        """Benchmark model performance with distributed data parallelism.

        Args:
            model: PyTorch model to benchmark.
            tokenizer: Tokenizer for the model.
            device: Device to run on ("cuda" or "cpu").
            model_name: Name of the model for reporting.
            batch_size: Total batch size (will be split across GPUs).
            tokens_per_request: Number of tokens per request for latency calculation.

        Returns:
            MeasuredMetrics with multi-GPU specific fields populated.

        Raises:
            RuntimeError: If distributed setup fails or required packages missing.
        """
        if self.gpu_count <= 1 or device == "cpu":
            # Run single GPU or CPU benchmark
            return self._benchmark_single_device(
                model, tokenizer, device, model_name, batch_size, tokens_per_request
            )

        # Multi-GPU benchmark with distributed context
        return self._benchmark_distributed(
            model, tokenizer, device, model_name, batch_size, tokens_per_request
        )

    def _benchmark_single_device(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        model_name: str,
        batch_size: int,
        tokens_per_request: int,
    ) -> MeasuredMetrics:
        """Benchmark on single GPU or CPU."""
        # Import deferred to avoid Windows CUDA deadlock
        import torch

        # Move model to device
        model = model.to(device)
        model.eval()

        if device == "cuda":
            torch.backends.cudnn.benchmark = True

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        parameters_b = param_count / 1e9

        # Prepare test input
        test_text = "The quick brown fox jumps over the lazy dog. " * 10
        inputs = tokenizer(test_text, return_tensors="pt").to(device)

        # Adjust batch size if needed
        if batch_size > 1:
            # Replicate inputs for batch dimension
            input_ids = inputs["input_ids"].repeat(batch_size, 1)
            attention_mask = inputs["attention_mask"].repeat(batch_size, 1)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Warmup
        with torch.inference_mode():
            for _ in range(self.warmup_iterations):
                _ = model(**inputs)

        # Measure throughput
        if device == "cuda":
            # Use CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]

            torch.cuda.synchronize()
            start_event.record()  # type: ignore[no-untyped-call]
            with torch.inference_mode():
                for _ in range(self.benchmark_iterations):
                    _ = model(**inputs)
            end_event.record()  # type: ignore[no-untyped-call]
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event) / 1000.0  # type: ignore[no-untyped-call] # convert ms to seconds
        else:
            # CPU timing
            start_time = time.time()
            with torch.inference_mode():
                for _ in range(self.benchmark_iterations):
                    _ = model(**inputs)
            total_time = time.time() - start_time

        avg_time = total_time / self.benchmark_iterations

        # Calculate throughput (tokens per second)
        num_tokens = inputs["input_ids"].shape[1] * batch_size
        throughput = (num_tokens * self.benchmark_iterations) / total_time

        # Get memory usage
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = model(**inputs)  # One forward pass to measure memory
            memory_bytes = torch.cuda.max_memory_allocated()
            memory_gb = memory_bytes / (1024**3)
            per_gpu_memory_gb = [memory_gb]  # Single GPU
        else:
            # Estimate for CPU
            memory_gb = param_count * 4 / (1024**3)  # 4 bytes per float32
            per_gpu_memory_gb = []

        latency_ms = avg_time * 1000

        # Store single GPU throughput for scaling efficiency calculation
        if self.gpu_count > 1 and self.measure_scaling_efficiency and device == "cuda":
            self._single_gpu_throughput = throughput

        return MeasuredMetrics(
            model_name=model_name,
            parameters_b=parameters_b,
            memory_gb=memory_gb,
            throughput_toks_per_sec=throughput,
            latency_ms_per_request=latency_ms,
            batch_size=batch_size,
            gpu_count=1,
            parallel_strategy=self.parallel_strategy,
            per_gpu_memory_gb=per_gpu_memory_gb,
        )

    def _benchmark_distributed(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        model_name: str,
        batch_size: int,
        tokens_per_request: int,
    ) -> MeasuredMetrics:
        """Benchmark with distributed data parallelism."""
        # Import deferred to avoid Windows CUDA deadlock
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        # Create distributed configuration
        self.distributed_config = DistributedConfig(
            distributed=True,
            num_gpus=self.gpu_count,
            parallel_strategy=self.parallel_strategy,
            distributed_backend=self.distributed_backend,
            distributed_init_method=self.distributed_init_method,
            batch_size_per_gpu=self.batch_size_per_gpu,
            warmup_iterations=self.warmup_iterations,
            benchmark_iterations=self.benchmark_iterations,
            measure_scaling_efficiency=self.measure_scaling_efficiency,
        )

        # Initialize distributed context
        with distributed_context(self.distributed_config) as ctx:
            rank = ctx.rank
            world_size = ctx.world_size
            local_rank = ctx.local_rank

            # Set device for this process
            torch.cuda.set_device(local_rank)

            # Move model to device
            model = model.to(local_rank)
            model.eval()

            # Wrap with DDP for data parallelism
            if self.parallel_strategy == "data":
                ddp_model = DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank
                )
                benchmark_model = ddp_model
            else:
                # For now, only data parallelism is implemented
                benchmark_model = model
                logger.warning(
                    "Only data parallelism is implemented. "
                    "Using model without parallel wrapper for strategy: %s",
                    self.parallel_strategy,
                )

            torch.backends.cudnn.benchmark = True

            # Count parameters (same on all ranks)
            param_count = sum(p.numel() for p in model.parameters())
            parameters_b = param_count / 1e9

            # Prepare test input - different data for each rank (data parallelism)
            test_text = f"The quick brown fox jumps over the lazy dog. Rank {rank}. " * 5
            inputs = tokenizer(test_text, return_tensors="pt").to(local_rank)

            # Adjust batch size per GPU
            effective_batch_size = self.batch_size_per_gpu
            if effective_batch_size > 1:
                input_ids = inputs["input_ids"].repeat(effective_batch_size, 1)
                attention_mask = inputs["attention_mask"].repeat(effective_batch_size, 1)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            # Warmup
            with torch.inference_mode():
                for _ in range(self.warmup_iterations):
                    _ = benchmark_model(**inputs)

            # Synchronize before timing
            dist.barrier()

            # Measure throughput with distributed timing
            start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]

            torch.cuda.synchronize()
            start_event.record()  # type: ignore[no-untyped-call]
            with torch.inference_mode():
                for _ in range(self.benchmark_iterations):
                    _ = benchmark_model(**inputs)
            end_event.record()  # type: ignore[no-untyped-call]
            torch.cuda.synchronize()

            # Synchronize across all processes
            dist.barrier()

            total_time = start_event.elapsed_time(end_event) / 1000.0  # type: ignore[no-untyped-call] # convert ms to seconds

            # Calculate local throughput
            num_tokens = inputs["input_ids"].shape[1] * effective_batch_size
            local_throughput = (num_tokens * self.benchmark_iterations) / total_time

            # Aggregate throughput across all processes
            throughput_tensor = torch.tensor([local_throughput], device=local_rank)
            synchronize_metric(throughput_tensor, reduce_op="sum")
            total_throughput = throughput_tensor.item()

            # Measure memory per GPU
            torch.cuda.reset_peak_memory_stats()
            _ = benchmark_model(**inputs)  # One forward pass
            memory_bytes = torch.cuda.max_memory_allocated()
            memory_gb = memory_bytes / (1024**3)

            # Gather memory usage from all GPUs
            memory_tensor = torch.tensor([memory_gb], device=local_rank)
            synchronize_metric(memory_tensor, reduce_op="sum")
            total_memory_gb = memory_tensor.item() / world_size  # Average per GPU

            # Collect per-GPU memory stats on rank 0
            per_gpu_memory_gb = []
            if rank == 0:
                # Gather from all ranks
                memory_list = [torch.zeros(1, device=local_rank) for _ in range(world_size)]
                dist.gather(
                    torch.tensor([memory_gb], device=local_rank),
                    gather_list=memory_list if rank == 0 else None,
                    dst=0,
                )
                if rank == 0:
                    per_gpu_memory_gb = [m.item() for m in memory_list]
            else:
                dist.gather(
                    torch.tensor([memory_gb], device=local_rank),
                    gather_list=None,
                    dst=0,
                )

            # Calculate scaling efficiency if single GPU baseline available
            scaling_efficiency = None
            if self.measure_scaling_efficiency and self._single_gpu_throughput is not None:
                ideal_throughput = self._single_gpu_throughput * world_size
                if ideal_throughput > 0:
                    scaling_efficiency = total_throughput / ideal_throughput

            # Estimate communication overhead (simplified)
            communication_overhead_ms = None
            if self.parallel_strategy == "data" and world_size > 1:
                # Simple estimation: time for all_reduce of small tensor
                test_tensor = torch.randn(100, 100, device=local_rank)
                torch.cuda.synchronize()
                start_time = time.time()
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                communication_overhead_ms = (time.time() - start_time) * 1000

            avg_time = total_time / self.benchmark_iterations
            latency_ms = avg_time * 1000

            # Only rank 0 returns full metrics
            if rank == 0:
                return MeasuredMetrics(
                    model_name=model_name,
                    parameters_b=parameters_b,
                    memory_gb=total_memory_gb,
                    throughput_toks_per_sec=total_throughput,
                    latency_ms_per_request=latency_ms,
                    batch_size=batch_size,
                    gpu_count=world_size,
                    parallel_strategy=self.parallel_strategy,
                    scaling_efficiency=scaling_efficiency,
                    communication_overhead_ms=communication_overhead_ms,
                    per_gpu_memory_gb=per_gpu_memory_gb,
                )
            else:
                # Non-main processes return dummy metrics (won't be used)
                return MeasuredMetrics(
                    model_name=model_name,
                    parameters_b=parameters_b,
                    memory_gb=0,
                    throughput_toks_per_sec=0,
                    latency_ms_per_request=0,
                    batch_size=batch_size,
                    gpu_count=world_size,
                )

    def compute_scaling_curve(
        self,
        model: Any,
        tokenizer: Any,
        gpu_counts: list[int],
        device: str = "cuda",
        **kwargs: Any,
    ) -> dict[int, dict[str, float | None]]:
        """Compute scaling efficiency curve across different GPU counts.

        Args:
            model: Model to benchmark.
            tokenizer: Tokenizer for the model.
            gpu_counts: List of GPU counts to test (e.g., [1, 2, 4, 8]).
            device: Device to use ("cuda").
            **kwargs: Additional arguments for benchmark_model.

        Returns:
            Dictionary mapping GPU count to metrics including throughput and
            scaling efficiency.
        """
        results = {}
        original_gpu_count = self.gpu_count

        try:
            for count in gpu_counts:
                self.gpu_count = count
                metrics = self.benchmark_model(model, tokenizer, device, **kwargs)
                results[count] = {
                    "throughput_toks_per_sec": metrics.throughput_toks_per_sec,
                    "memory_gb": metrics.memory_gb,
                    "scaling_efficiency": metrics.scaling_efficiency,
                    "gpu_count": metrics.gpu_count,
                }
        finally:
            self.gpu_count = original_gpu_count

        return results
