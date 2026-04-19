from dotenv import load_dotenv
import os
from typing import Optional
from dataclasses import dataclass, field

load_dotenv()

@dataclass
class Config:
    """Configuration for Qwen Inference Engine"""

    # Model
    model_path: str = field(default_factory=lambda: os.getenv('MODEL_PATH', ''))

    # GPU
    cuda_visible_devices: str = field(default_factory=lambda: os.getenv('CUDA_VISIBLE_DEVICES', '2,3'))
    tensor_parallel_size: int = field(default_factory=lambda: int(os.getenv('TENSOR_PARALLEL_SIZE', '2')))

    # Memory
    max_gpu_memory_gb: int = field(default_factory=lambda: int(os.getenv('MAX_GPU_MEMORY_GB', '48')))
    kv_cache_block_size: int = field(default_factory=lambda: int(os.getenv('KV_CACHE_BLOCK_SIZE', '16')))
    max_sequence_length: int = field(default_factory=lambda: int(os.getenv('MAX_SEQUENCE_LENGTH', '250000')))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv('MAX_BATCH_SIZE', '4')))

    # Paged Attention
    block_size: int = field(default_factory=lambda: int(os.getenv('BLOCK_SIZE', '16')))
    num_gpu_blocks: int = field(default_factory=lambda: int(os.getenv('NUM_GPU_BLOCKS', '120000')))
    num_cpu_blocks: int = field(default_factory=lambda: int(os.getenv('NUM_CPU_BLOCKS', '10000')))

    # GPTQ
    gptq_bits: int = field(default_factory=lambda: int(os.getenv('GPTQ_BITS', '4')))
    gptq_groupsize: int = field(default_factory=lambda: int(os.getenv('GPTQ_GROUPSIZE', '128')))

    # Safety
    safety_margin_gb: float = field(default_factory=lambda: float(os.getenv('SAFETY_MARGIN_GB', '2.0')))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_memory_usage: bool = field(default_factory=lambda: os.getenv('LOG_MEMORY_USAGE', 'true').lower() == 'true')

    @property
    def gpu_indices(self) -> list[int]:
        """Parse CUDA_VISIBLE_DEVICES into list of integers"""
        return [int(x.strip()) for x in self.cuda_visible_devices.split(',')]

    def validate(self) -> None:
        """Validate configuration"""
        if not self.model_path:
            raise ValueError("MODEL_PATH must be set in .env file")

        if self.tensor_parallel_size != len(self.gpu_indices):
            raise ValueError(
                f"TENSOR_PARALLEL_SIZE ({self.tensor_parallel_size}) must match "
                f"number of GPUs in CUDA_VISIBLE_DEVICES ({len(self.gpu_indices)})"
            )