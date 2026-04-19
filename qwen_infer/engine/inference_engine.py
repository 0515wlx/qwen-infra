"""
Main Inference Engine

High-level interface for Qwen model inference.
Coordinates memory management, tensor parallelism, and long sequence handling.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Optional, Dict, Generator, Tuple
from pathlib import Path
import time

from qwen_infer.config.settings import Config
from qwen_infer.memory.memory_manager import MultiGPUMemoryManager
from qwen_infer.attention.paged_attention import PagedAttention
from qwen_infer.models.gptq_loader import GPTQModelLoader, GPTQConfig
from qwen_infer.utils.memory_utils import setup_logging, log_memory_usage

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Qwen Inference Engine

    Manages the complete inference pipeline including:
    - Model loading and initialization
    - Multi-GPU tensor parallelism
    - Paged attention with KV cache management
    - Long sequence support (>200k tokens)
    - Strict memory management
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.validate()

        # Setup
        setup_logging(self.config.log_level)
        self.gpu_ids = self.config.gpu_indices
        self.num_gpus = len(self.gpu_ids)

        logger.info(f"Initializing Qwen Inference Engine")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Using GPUs: {self.gpu_ids}")

        # Memory manager
        self.memory_manager = MultiGPUMemoryManager(
            gpu_ids=self.gpu_ids,
            total_memory_per_gpu_gb=self.config.max_gpu_memory_gb,
            safety_margin_gb=self.config.safety_margin_gb
        )

        # Paged attention
        self.paged_attention: Optional[PagedAttention] = None

        # Model
        self.model = None
        self.model_config = None
        self.tokenizer = None

        # Statistics
        self.stats = {
            'requests_processed': 0,
            'tokens_generated': 0,
            'peak_memory_gb': 0.0,
            'avg_latency_ms': 0.0
        }

        # Sequence tracking
        self.active_sequences: Dict[int, Dict] = {}
        self.next_seq_id = 0

    def initialize(self) -> bool:
        """
        Initialize the inference engine.

        Returns:
            bool: True if initialization successful
        """
        try:
            # Set CUDA devices
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = self.config.cuda_visible_devices

            # Validate GPUs
            if not torch.cuda.is_available():
                logger.error("CUDA not available")
                return False

            available_gpus = torch.cuda.device_count()
            logger.info(f"Available CUDA devices: {available_gpus}")

            for gpu_id in self.gpu_ids:
                if gpu_id >= available_gpus:
                    logger.error(f"GPU {gpu_id} not available (only {available_gpus} GPUs found)")
                    return False

            # Initialize memory managers
            for gpu_id in self.gpu_ids:
                torch.cuda.set_device(gpu_id)
                torch.cuda.synchronize()
                logger.info(f"Initialized GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

            # Log initial memory state
            if self.config.log_memory_usage:
                log_memory_usage(self.gpu_ids, "Initial: ")

            # Load model
            if not self._load_model():
                return False

            # Initialize paged attention
            self._initialize_paged_attention()

            # Preallocate memory pools
            self._preallocate_memory()

            logger.info("Inference engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def _load_model(self) -> bool:
        """Load the GPTQ quantized model"""
        logger.info(f"Loading model from {self.config.model_path}")

        try:
            loader = GPTQModelLoader(
                self.config.model_path,
                quant_config=GPTQConfig(
                    bits=self.config.gptq_bits,
                    group_size=self.config.gptq_groupsize
                )
            )

            # Load config and estimate memory
            model_config = loader.load_config()
            mem_estimate = loader.estimate_memory_usage()

            logger.info(f"Model config: {model_config}")
            logger.info(f"Estimated memory: {mem_estimate}")

            # Verify we have enough memory
            status = self.memory_manager.get_balanced_memory_status()
            if mem_estimate['per_gpu_gb'] > status.free_gb / self.num_gpus:
                logger.warning(
                    f"Estimated per-GPU memory ({mem_estimate['per_gpu_gb']:.2f}GB) "
                    f"exceeds available memory ({status.free_gb / self.num_gpus:.2f}GB)"
                )

            # Load weights (placeholder for now)
            self.model = loader.load_model_for_gpus(self.gpu_ids)
            self.model_config = model_config

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return False

    def _initialize_paged_attention(self) -> None:
        """Initialize paged attention system"""
        if self.model_config is None:
            raise RuntimeError("Model not loaded")

        num_heads = self.model_config.get('num_attention_heads', 28)
        head_dim = self.model_config.get('head_dim', 128)

        self.paged_attention = PagedAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=self.config.block_size,
            num_blocks=self.config.num_gpu_blocks,
            device=torch.device(f'cuda:{self.gpu_ids[0]}')
        )

        logger.info(
            f"Initialized paged attention: {self.config.num_gpu_blocks} blocks, "
            f"block_size={self.config.block_size}"
        )

    def _preallocate_memory(self) -> None:
        """Preallocate memory pools for KV cache"""
        logger.info("Preallocating memory pools...")

        # Calculate pool sizes based on model config
        if self.model_config:
            num_layers = self.model_config.get('num_hidden_layers', 60)
            num_kv_heads = self.model_config.get('num_key_value_heads', 4)
            head_dim = self.model_config.get('head_dim', 128)

            # KV cache pool size
            elements_per_block = (
                self.config.block_size * num_kv_heads * head_dim * 2  # K and V
            )
            bytes_per_block = elements_per_block * 2  # fp16
            total_bytes = bytes_per_block * self.config.num_gpu_blocks

            # Preallocate on each GPU
            pool_configs = [
                ('kv_cache', total_bytes, torch.float16)
            ]

            if not self.memory_manager.preallocate_pools_all(pool_configs):
                logger.warning("Failed to preallocate all memory pools")

        if self.config.log_memory_usage:
            log_memory_usage(self.gpu_ids, "After preallocation: ")

    def create_sequence(self, prompt_tokens: List[int]) -> int:
        """
        Create a new inference sequence.

        Args:
            prompt_tokens: Initial prompt token IDs

        Returns:
            int: Sequence ID
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1

        # Allocate blocks for sequence
        if not self.paged_attention.allocate_sequence(seq_id, len(prompt_tokens)):
            raise RuntimeError(f"Failed to allocate memory for sequence {seq_id}")

        self.active_sequences[seq_id] = {
            'tokens': prompt_tokens.copy(),
            'length': len(prompt_tokens),
            'generated': 0,
            'gpu_id': self.gpu_ids[seq_id % len(self.gpu_ids)]
        }

        logger.debug(f"Created sequence {seq_id} with {len(prompt_tokens)} tokens")
        return seq_id

    def generate(
        self,
        seq_id: int,
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_tokens: Optional[List[int]] = None
    ) -> Generator[int, None, None]:
        """
        Generate tokens for a sequence.

        Args:
            seq_id: Sequence ID
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_tokens: List of token IDs that stop generation

        Yields:
            int: Generated token IDs
        """
        if seq_id not in self.active_sequences:
            raise ValueError(f"Sequence {seq_id} not found")

        seq_info = self.active_sequences[seq_id]

        for i in range(max_new_tokens):
            # Check memory pressure
            status = self.memory_manager.get_balanced_memory_status()
            if status.pressure_level.value == 'eviction':
                logger.warning("Memory pressure critical, performing emergency cleanup")
                self.memory_manager.emergency_cleanup_all()

            # Extend sequence if needed
            if not self.paged_attention.append_tokens(seq_id, 1):
                logger.error(f"Failed to extend sequence {seq_id}")
                break

            # TODO: Run actual inference
            # For now, return dummy token
            next_token = 0  # Placeholder

            # Check stop tokens
            if stop_tokens and next_token in stop_tokens:
                break

            # Update sequence
            seq_info['tokens'].append(next_token)
            seq_info['length'] += 1
            seq_info['generated'] += 1

            self.stats['tokens_generated'] += 1

            yield next_token

        self.stats['requests_processed'] += 1

    def get_sequence_length(self, seq_id: int) -> int:
        """Get current length of a sequence"""
        if seq_id not in self.active_sequences:
            return 0
        return self.active_sequences[seq_id]['length']

    def free_sequence(self, seq_id: int) -> None:
        """Free a sequence and its allocated memory"""
        if seq_id in self.active_sequences:
            self.paged_attention.free_sequence(seq_id)
            del self.active_sequences[seq_id]
            logger.debug(f"Freed sequence {seq_id}")

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        status = self.memory_manager.get_balanced_memory_status()

        return {
            **self.stats,
            'memory_status': {
                'allocated_gb': status.allocated_gb,
                'free_gb': status.free_gb,
                'pressure_level': status.pressure_level.value
            },
            'active_sequences': len(self.active_sequences),
            'paged_attention': self.paged_attention.get_memory_stats() if self.paged_attention else None
        }

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        if self.paged_attention:
            for seq_id in list(self.active_sequences.keys()):
                self.free_sequence(seq_id)

        torch.cuda.empty_cache()
        logger.info("Inference engine shutdown")