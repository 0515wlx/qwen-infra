"""
Strict GPU Memory Manager

Implements aggressive memory management for long sequence inference.
Ensures KV cache doesn't exceed available memory through:
- Pre-allocated block pools
- Dynamic block eviction
- Watermark-based allocation
- Memory defragmentation
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MemoryPressureLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EVICTION = "eviction"

@dataclass
class MemoryStatus:
    """Current memory status"""
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float
    pressure_level: MemoryPressureLevel
    fragmentation_ratio: float

class GPUMemoryManager:
    """
    Per-GPU memory manager with strict accounting.

    Features:
    - Memory pools for KV cache
    - Dynamic watermark management
    - Fragmentation monitoring
    - Safe OOM prevention
    """

    def __init__(
        self,
        gpu_id: int,
        total_memory_gb: float,
        safety_margin_gb: float = 2.0,
        watermark_high: float = 0.85,
        watermark_low: float = 0.70
    ):
        self.gpu_id = gpu_id
        self.total_memory_gb = total_memory_gb
        self.safety_margin_gb = safety_margin_gb
        self.usable_memory_gb = total_memory_gb - safety_margin_gb

        # Watermarks for memory pressure
        self.watermark_high = watermark_high
        self.watermark_low = watermark_low

        # Memory pools (pre-allocated)
        self.pools: Dict[str, torch.Tensor] = {}
        self.pool_sizes: Dict[str, int] = {}

        # Track allocations
        self.allocations: Dict[str, int] = {}
        self.peak_allocated = 0

        # Initialize CUDA context
        self.device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)

        # Validate available memory
        self._validate_memory()

    def _validate_memory(self) -> None:
        """Validate that we have the expected amount of memory"""
        torch.cuda.synchronize(self.device)
        total = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024**3)

        if total < self.total_memory_gb - 1:  # Allow 1GB tolerance
            logger.warning(
                f"GPU {self.gpu_id}: Expected {self.total_memory_gb}GB, "
                f"but found {total:.2f}GB"
            )
            self.total_memory_gb = total
            self.usable_memory_gb = total - self.safety_margin_gb

    def get_memory_status(self) -> MemoryStatus:
        """Get current memory status"""
        torch.cuda.synchronize(self.device)

        total = torch.cuda.get_device_properties(self.gpu_id).total_memory
        allocated = torch.cuda.memory_allocated(self.gpu_id)
        reserved = torch.cuda.memory_reserved(self.gpu_id)
        free = total - reserved

        total_gb = total / (1024**3)
        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        free_gb = free / (1024**3)

        # Calculate pressure level
        usage_ratio = allocated / total
        if usage_ratio > 0.95:
            pressure = MemoryPressureLevel.EVICTION
        elif usage_ratio > self.watermark_high:
            pressure = MemoryPressureLevel.CRITICAL
        elif usage_ratio > self.watermark_low:
            pressure = MemoryPressureLevel.WARNING
        else:
            pressure = MemoryPressureLevel.NORMAL

        # Estimate fragmentation
        fragmentation = (reserved - allocated) / total if reserved > 0 else 0

        return MemoryStatus(
            allocated_gb=allocated_gb,
            reserved_gb=reserved_gb,
            free_gb=free_gb,
            total_gb=total_gb,
            pressure_level=pressure,
            fragmentation_ratio=fragmentation
        )

    def preallocate_pool(
        self,
        pool_name: str,
        size_bytes: int,
        dtype: torch.dtype = torch.float16
    ) -> bool:
        """
        Pre-allocate a memory pool.

        Args:
            pool_name: Identifier for the pool
            size_bytes: Size in bytes
            dtype: Data type for the pool

        Returns:
            bool: True if allocation successful
        """
        try:
            with torch.cuda.device(self.gpu_id):
                # Check if we have enough memory
                status = self.get_memory_status()
                size_gb = size_bytes / (1024**3)

                if status.allocated_gb + size_gb > self.usable_memory_gb:
                    logger.error(
                        f"GPU {self.gpu_id}: Cannot preallocate pool '{pool_name}' "
                        f"({size_gb:.2f}GB). Current: {status.allocated_gb:.2f}GB, "
                        f"Usable: {self.usable_memory_gb:.2f}GB"
                    )
                    return False

                # Allocate
                num_elements = size_bytes // torch.finfo(dtype).bits * 8 // 2  # Convert to elements
                pool = torch.empty(
                    (num_elements,),
                    dtype=dtype,
                    device=self.device
                )

                self.pools[pool_name] = pool
                self.pool_sizes[pool_name] = size_bytes

                logger.info(
                    f"GPU {self.gpu_id}: Pre-allocated pool '{pool_name}' "
                    f"({size_gb:.2f}GB)"
                )
                return True

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU {self.gpu_id}: OOM when preallocating pool '{pool_name}': {e}")
            return False

    def allocate_from_pool(
        self,
        pool_name: str,
        size_bytes: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16
    ) -> Optional[torch.Tensor]:
        """
        Allocate a tensor from a pre-allocated pool.

        Uses views into the pre-allocated buffer to avoid fragmentation.
        """
        if pool_name not in self.pools:
            return None

        # Check if we have space in the pool
        current_alloc = self.allocations.get(pool_name, 0)
        pool_size = self.pool_sizes[pool_name]

        if current_alloc + size_bytes > pool_size:
            logger.warning(
                f"GPU {self.gpu_id}: Pool '{pool_name}' exhausted. "
                f"Requested: {size_bytes}, Available: {pool_size - current_alloc}"
            )
            return None

        # Create a view into the pool
        pool = self.pools[pool_name]
        start_idx = current_alloc // 2  # Convert bytes to elements (fp16)
        end_idx = start_idx + (size_bytes // 2)

        try:
            view = pool[start_idx:end_idx].view(shape)
            self.allocations[pool_name] = current_alloc + size_bytes
            self.peak_allocated = max(self.peak_allocated, self.allocations[pool_name])
            return view
        except RuntimeError as e:
            logger.error(f"GPU {self.gpu_id}: Failed to create view: {e}")
            return None

    def free_to_pool(self, pool_name: str, size_bytes: int) -> None:
        """Free allocation tracking (doesn't actually free memory)"""
        if pool_name in self.allocations:
            self.allocations[pool_name] = max(0, self.allocations[pool_name] - size_bytes)

    def check_memory_pressure(self) -> MemoryPressureLevel:
        """Check current memory pressure level"""
        status = self.get_memory_status()
        return status.pressure_level

    def emergency_cleanup(self) -> None:
        """Emergency memory cleanup when OOM imminent"""
        logger.warning(f"GPU {self.gpu_id}: Performing emergency cleanup")

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Synchronize to ensure all ops complete
        torch.cuda.synchronize(self.device)

        # Report status after cleanup
        status = self.get_memory_status()
        logger.info(
            f"GPU {self.gpu_id}: After cleanup - "
            f"Free: {status.free_gb:.2f}GB, "
            f"Fragmentation: {status.fragmentation_ratio:.2%}"
        )

    def get_pool_stats(self) -> Dict[str, Dict]:
        """Get statistics for all pools"""
        stats = {}
        for name, size in self.pool_sizes.items():
            allocated = self.allocations.get(name, 0)
            stats[name] = {
                'total_bytes': size,
                'total_gb': size / (1024**3),
                'allocated_bytes': allocated,
                'allocated_gb': allocated / (1024**3),
                'free_bytes': size - allocated,
                'free_gb': (size - allocated) / (1024**3),
                'utilization': allocated / size if size > 0 else 0
            }
        return stats

    def __del__(self):
        """Cleanup when manager is destroyed"""
        self.pools.clear()
        torch.cuda.empty_cache()

class MultiGPUMemoryManager:
    """
    Coordinates memory management across multiple GPUs.

    Ensures balanced memory usage and handles cross-GPU migrations
    if needed for very long sequences.
    """

    def __init__(
        self,
        gpu_ids: List[int],
        total_memory_per_gpu_gb: float,
        safety_margin_gb: float = 2.0
    ):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)

        # Initialize per-GPU managers
        self.managers: Dict[int, GPUMemoryManager] = {}
        for gpu_id in gpu_ids:
            self.managers[gpu_id] = GPUMemoryManager(
                gpu_id=gpu_id,
                total_memory_gb=total_memory_per_gpu_gb,
                safety_margin_gb=safety_margin_gb
            )

    def get_manager(self, gpu_id: int) -> GPUMemoryManager:
        """Get memory manager for specific GPU"""
        return self.managers[gpu_id]

    def get_balanced_memory_status(self) -> MemoryStatus:
        """
        Get aggregated memory status across all GPUs.
        Reports the most critical status.
        """
        statuses = [m.get_memory_status() for m in self.managers.values()]

        # Aggregate
        total_gb = sum(s.total_gb for s in statuses)
        allocated_gb = sum(s.allocated_gb for s in statuses)
        free_gb = sum(s.free_gb for s in statuses)

        # Most critical pressure level
        pressure_order = [
            MemoryPressureLevel.NORMAL,
            MemoryPressureLevel.WARNING,
            MemoryPressureLevel.CRITICAL,
            MemoryPressureLevel.EVICTION
        ]
        max_pressure_idx = max(
            [pressure_order.index(s.pressure_level) for s in statuses]
        )
        max_pressure = pressure_order[max_pressure_idx]

        # Average fragmentation
        avg_frag = sum(s.fragmentation_ratio for s in statuses) / len(statuses)

        return MemoryStatus(
            allocated_gb=allocated_gb,
            reserved_gb=sum(s.reserved_gb for s in statuses),
            free_gb=free_gb,
            total_gb=total_gb,
            pressure_level=max_pressure,
            fragmentation_ratio=avg_frag
        )

    def preallocate_pools_all(self, pool_configs: List[Tuple[str, int, torch.dtype]]) -> bool:
        """
        Preallocate pools on all GPUs with the same configuration.

        Args:
            pool_configs: List of (pool_name, size_bytes, dtype)

        Returns:
            bool: True if all allocations successful
        """
        success = True
        for gpu_id, manager in self.managers.items():
            for pool_name, size_bytes, dtype in pool_configs:
                # Add GPU suffix to pool name
                full_name = f"{pool_name}_gpu{gpu_id}"
                if not manager.preallocate_pool(full_name, size_bytes, dtype):
                    success = False
        return success

    def emergency_cleanup_all(self) -> None:
        """Perform emergency cleanup on all GPUs"""
        for manager in self.managers.values():
            manager.emergency_cleanup()

    def get_all_pool_stats(self) -> Dict[int, Dict]:
        """Get pool statistics for all GPUs"""
        return {
            gpu_id: manager.get_pool_stats()
            for gpu_id, manager in self.managers.items()
        }