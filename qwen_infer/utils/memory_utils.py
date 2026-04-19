"""Utility functions for Qwen Inference Engine"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_gpu_memory_info(gpu_id: int) -> dict:
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return {'total': 0, 'allocated': 0, 'free': 0}

    torch.cuda.set_device(gpu_id)
    total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
    reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
    free = total - reserved

    return {
        'total': total,
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'gpu_id': gpu_id
    }

def log_memory_usage(gpu_ids: list[int], prefix: str = "") -> None:
    """Log memory usage for specified GPUs"""
    for gpu_id in gpu_ids:
        info = get_gpu_memory_info(gpu_id)
        logger.info(
            f"{prefix}GPU {gpu_id}: Total={info['total']:.2f}GB, "
            f"Allocated={info['allocated']:.2f}GB, "
            f"Reserved={info['reserved']:.2f}GB, Free={info['free']:.2f}GB"
        )

def calculate_num_blocks(
    available_memory_gb: float,
    block_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,  # fp16
    safety_margin_gb: float = 2.0
) -> int:
    """
    Calculate number of KV cache blocks that can fit in available memory

    Each block contains:
    - Key cache: [block_size, num_heads, head_dim]
    - Value cache: [block_size, num_heads, head_dim]
    """
    # Calculate memory per block
    bytes_per_element = dtype_bytes
    elements_per_block = block_size * num_heads * head_dim * 2  # K + V
    memory_per_block = elements_per_block * bytes_per_element
    memory_per_block_gb = memory_per_block / (1024**3)

    # Account for safety margin
    usable_memory_gb = available_memory_gb - safety_margin_gb

    # Calculate number of blocks
    num_blocks = int(usable_memory_gb / memory_per_block_gb)

    return max(num_blocks, 0)

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False
) -> list[torch.Tensor]:
    """Split a tensor along its last dimension."""
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return [t.contiguous() for t in tensor_list]
    return list(tensor_list)

class CUDAAllocator:
    """CUDA memory allocator with tracking"""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.allocated = 0

    def allocate(self, size_bytes: int) -> torch.Tensor:
        """Allocate memory and track usage"""
        with torch.cuda.device(self.device_id):
            tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device='cuda')
            self.allocated += size_bytes
            return tensor

    def get_allocated_memory(self) -> int:
        """Get total allocated memory in bytes"""
        return self.allocated