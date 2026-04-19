"""
KV Cache Manager

Implements fine-grained KV cache management for long sequences.
Handles block allocation, eviction, and migration.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class KVBlock:
    """Represents a single KV cache block"""
    block_id: int
    gpu_id: int
    start_token: int
    end_token: int
    ref_count: int = 0
    last_accessed: float = 0.0

    def contains_token(self, token_idx: int) -> bool:
        return self.start_token <= token_idx < self.end_token

    def __repr__(self):
        return f"KVBlock(id={self.block_id}, gpu={self.gpu_id}, tokens={self.start_token}-{self.end_token}, refs={self.ref_count})"

class KVCacheManager:
    """
    Fine-grained KV Cache Manager for long sequences.

    Features:
    - Block-based storage with variable sizes
    - LRU eviction policy
    - GPU/CPU migration support
    - Memory-bounded allocation
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks: int,
        gpu_ids: List[int],
        dtype: torch.dtype = torch.float16
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype

        # Track blocks per sequence per layer
        self.seq_blocks: Dict[int, Dict[int, List[KVBlock]]] = {}  # seq_id -> layer -> blocks
        self.seq_lengths: Dict[int, int] = {}  # seq_id -> length

        # Global block pool
        self.blocks: Dict[int, KVBlock] = {}
        self.free_blocks: List[int] = []
        self.next_block_id = 0

        # Initialize free block pool
        for _ in range(max_blocks):
            self.free_blocks.append(self.next_block_id)
            self.next_block_id += 1

        # Per-GPU storage
        self.gpu_storage: Dict[int, Dict[int, torch.Tensor]] = {}  # gpu_id -> layer -> tensor
        for gpu_id in gpu_ids:
            self.gpu_storage[gpu_id] = {}
            # Each layer needs KV storage: [num_blocks, block_size, num_heads, head_dim]
            shape = (max_blocks, block_size, num_heads, head_dim)
            try:
                with torch.cuda.device(gpu_id):
                    self.gpu_storage[gpu_id] = {
                        'k': torch.empty(shape, dtype=dtype, device=gpu_id),
                        'v': torch.empty(shape, dtype=dtype, device=gpu_id)
                    }
                    logger.info(f"GPU {gpu_id}: Allocated KV cache storage ({max_blocks} blocks)")
            except torch.cuda.OutOfMemoryError:
                logger.error(f"GPU {gpu_id}: Failed to allocate KV cache storage")
                raise

        # LRU tracking
        self.access_order: OrderedDict[int, float] = OrderedDict()

        # Statistics
        self.stats = {
            'total_blocks': max_blocks,
            'allocated_blocks': 0,
            'evictions': 0,
            'hits': 0,
            'misses': 0
        }

    def allocate_sequence(self, seq_id: int, initial_length: int, gpu_id: int) -> bool:
        """
        Allocate blocks for a new sequence.

        Args:
            seq_id: Sequence identifier
            initial_length: Initial token count
            gpu_id: Target GPU

        Returns:
            bool: True if allocation successful
        """
        if seq_id in self.seq_blocks:
            return True  # Already allocated

        num_blocks_needed = (initial_length + self.block_size - 1) // self.block_size

        # Try to allocate blocks
        blocks = []
        for _ in range(num_blocks_needed):
            if not self.free_blocks:
                # Try to evict LRU blocks
                if not self._evict_lru_block():
                    logger.error(f"Cannot allocate sequence {seq_id}: no free blocks and eviction failed")
                    return False
                break

            block_id = self.free_blocks.pop()
            start_token = len(blocks) * self.block_size
            block = KVBlock(
                block_id=block_id,
                gpu_id=gpu_id,
                start_token=start_token,
                end_token=start_token + self.block_size,
                ref_count=1
            )
            self.blocks[block_id] = block
            blocks.append(block)

        if len(blocks) < num_blocks_needed:
            # Rollback partial allocation
            for block in blocks:
                self.free_blocks.append(block.block_id)
                del self.blocks[block.block_id]
            return False

        # Initialize layer tracking
        self.seq_blocks[seq_id] = {layer: list(blocks) for layer in range(self.num_layers)}
        self.seq_lengths[seq_id] = initial_length
        self.stats['allocated_blocks'] += len(blocks)

        logger.debug(f"Allocated {len(blocks)} blocks for sequence {seq_id}")
        return True

    def extend_sequence(self, seq_id: int, new_length: int) -> bool:
        """
        Extend sequence with more tokens, allocating new blocks if needed.

        Args:
            seq_id: Sequence identifier
            new_length: New total length

        Returns:
            bool: True if extension successful
        """
        if seq_id not in seq_blocks:
            return False

        current_length = self.seq_lengths[seq_id]
        if new_length <= current_length:
            return True

        current_blocks = self.seq_blocks[seq_id][0]  # Use layer 0 as reference
        needed_blocks = (new_length + self.block_size - 1) // self.block_size
        current_block_count = len(current_blocks)

        if needed_blocks <= current_block_count:
            # Existing blocks can cover the new length
            self.seq_lengths[seq_id] = new_length
            return True

        # Need to allocate more blocks
        additional_needed = needed_blocks - current_block_count

        for i in range(additional_needed):
            if not self.free_blocks:
                if not self._evict_lru_block():
                    return False

            block_id = self.free_blocks.pop()
            start_token = (current_block_count + i) * self.block_size
            gpu_id = current_blocks[0].gpu_id  # Use same GPU

            block = KVBlock(
                block_id=block_id,
                gpu_id=gpu_id,
                start_token=start_token,
                end_token=start_token + self.block_size,
                ref_count=1
            )

            self.blocks[block_id] = block
            for layer in range(self.num_layers):
                self.seq_blocks[seq_id][layer].append(block)
            self.stats['allocated_blocks'] += 1

        self.seq_lengths[seq_id] = new_length
        return True

    def _evict_lru_block(self) -> bool:
        """
        Evict the least recently used block that's not referenced.

        Returns:
            bool: True if eviction successful
        """
        # Find LRU block with ref_count == 0
        candidates = [
            (block_id, block)
            for block_id, block in self.blocks.items()
            if block.ref_count == 0
        ]

        if not candidates:
            logger.warning("Cannot evict: all blocks have active references")
            return False

        # Sort by last_accessed (oldest first)
        candidates.sort(key=lambda x: x[1].last_accessed)

        block_id, block = candidates[0]

        # Remove from sequence tracking
        for seq_id, layers in self.seq_blocks.items():
            for layer, blocks in layers.items():
                if block in blocks:
                    blocks.remove(block)
                    break

        # Return to free pool
        del self.blocks[block_id]
        self.free_blocks.append(block_id)
        self.stats['evictions'] += 1
        self.stats['allocated_blocks'] -= 1

        logger.debug(f"Evicted block {block_id}")
        return True

    def get_kv_tensors(
        self,
        seq_id: int,
        layer: int,
        start_token: int,
        end_token: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get KV tensors for a token range.

        Args:
            seq_id: Sequence identifier
            layer: Layer index
            start_token: Start token position (inclusive)
            end_token: End token position (exclusive)

        Returns:
            Tuple of (K, V) tensors, or None if not cached
        """
        if seq_id not in self.seq_blocks:
            self.stats['misses'] += 1
            return None

        if layer not in self.seq_blocks[seq_id]:
            self.stats['misses'] += 1
            return None

        blocks = self.seq_blocks[seq_id][layer]

        # Find blocks covering the range
        k_parts = []
        v_parts = []

        for block in blocks:
            if block.end_token <= start_token:
                continue
            if block.start_token >= end_token:
                break

            # Calculate overlap
            overlap_start = max(start_token, block.start_token)
            overlap_end = min(end_token, block.end_token)
            tokens_in_block = overlap_end - overlap_start

            if tokens_in_block <= 0:
                continue

            # Get tensor slice
            offset_in_block = overlap_start - block.start_token
            gpu_id = block.gpu_id

            k_block = self.gpu_storage[gpu_id]['k'][block.block_id]
            v_block = self.gpu_storage[gpu_id]['v'][block.block_id]

            # Slice to valid tokens
            k_parts.append(k_block[offset_in_block:offset_in_block + tokens_in_block])
            v_parts.append(v_block[offset_in_block:offset_in_block + tokens_in_block])

            # Update access time and ref count
            block.last_accessed = torch.cuda.Event(enable_timing=True).elapsed_time
            block.add_ref()

        if not k_parts:
            self.stats['misses'] += 1
            return None

        self.stats['hits'] += 1
        k = torch.cat(k_parts, dim=0)
        v = torch.cat(v_parts, dim=0)

        return k, v

    def release_blocks(self, seq_id: int, layer: int) -> None:
        """Release reference to blocks for a layer after attention computation"""
        if seq_id not in self.seq_blocks:
            return
        if layer not in self.seq_blocks[seq_id]:
            return

        for block in self.seq_blocks[seq_id][layer]:
            block.remove_ref()

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a sequence"""
        if seq_id not in self.seq_blocks:
            return

        for layer, blocks in self.seq_blocks[seq_id].items():
            for block in blocks:
                block.ref_count = 0
                del self.blocks[block.block_id]
                self.free_blocks.append(block.block_id)

        del self.seq_blocks[seq_id]
        del self.seq_lengths[seq_id]

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        hit_rate = (
            self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
            if (self.stats['hits'] + self.stats['misses']) > 0
            else 0.0
        )

        return {
            **self.stats,
            'free_blocks': len(self.free_blocks),
            'hit_rate': hit_rate
        }