"""
Paged Attention Mechanism

Implements block-based KV cache management similar to vLLM.
Supports efficient memory reuse for variable-length sequences.
"""

import torch
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class Block:
    """A block in the paged attention system"""
    block_number: int
    device: torch.device
    ref_count: int = 0

    def __post_init__(self):
        self.ref_count = 0

    def add_ref(self) -> None:
        self.ref_count += 1

    def remove_ref(self) -> None:
        assert self.ref_count > 0, "Removing reference from block with 0 refs"
        self.ref_count -= 1

class BlockAllocator:
    """
    Manages allocation and deallocation of KV cache blocks.

    Similar to vLLM's block allocator, but with stricter memory tracking
    for long sequence support (>200k tokens).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Calculate block memory requirement
        elements_per_block = block_size * num_heads * head_dim * 2  # K and V
        bytes_per_block = elements_per_block * torch.finfo(dtype).bits // 8

        # Pre-allocate all blocks
        self.k_cache = torch.empty(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.empty(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )

        # Track block allocation
        self.free_blocks = set(range(num_blocks))
        self.block_ref_count: Dict[int, int] = {i: 0 for i in range(num_blocks)}

        # Statistics
        self.num_allocated = 0
        self.num_free = num_blocks

    def allocate_block(self) -> Optional[int]:
        """Allocate a single block"""
        if not self.free_blocks:
            return None

        block_num = self.free_blocks.pop()
        self.block_ref_count[block_num] = 1
        self.num_allocated += 1
        self.num_free = len(self.free_blocks)
        return block_num

    def allocate_blocks(self, num_blocks: int) -> List[int]:
        """Allocate multiple blocks"""
        allocated = []
        for _ in range(num_blocks):
            block_num = self.allocate_block()
            if block_num is None:
                # Rollback allocations
                for bn in allocated:
                    self.free_block(bn)
                return []
            allocated.append(block_num)
        return allocated

    def free_block(self, block_num: int) -> None:
        """Free a block back to the pool"""
        assert self.block_ref_count[block_num] > 0, "Freeing unallocated block"
        self.block_ref_count[block_num] -= 1

        if self.block_ref_count[block_num] == 0:
            self.free_blocks.add(block_num)
            self.num_allocated -= 1
            self.num_free = len(self.free_blocks)

    def fork_block(self, block_num: int) -> int:
        """Create a copy-on-write reference to a block"""
        assert self.block_ref_count[block_num] > 0, "Forking unallocated block"
        self.block_ref_count[block_num] += 1
        return block_num

    def get_kv_cache(self, block_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV cache tensors for a specific block"""
        return self.k_cache[block_num], self.v_cache[block_num]

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        return {
            'allocated_blocks': self.num_allocated,
            'free_blocks': self.num_free,
            'total_blocks': self.num_blocks,
            'allocated_gb': self.num_allocated * self.block_size * self.num_heads * self.head_dim * 2 * 2 / (1024**3)
        }

class PagedAttention:
    """
    Paged Attention implementation supporting long sequences.

    Key features:
    - Block-based KV cache management
    - Efficient memory sharing between sequences
    - Support for >200k token sequences through aggressive memory management
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
        block_size: int = 16,
        num_blocks: int = 100000,
        device: torch.device = torch.device('cuda')
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale or 1.0 / math.sqrt(head_dim)
        self.block_size = block_size
        self.device = device

        # Initialize block allocator
        self.block_allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device
        )

        # Track sequence block mappings
        self.seq_blocks: Dict[int, List[int]] = {}
        self.seq_lengths: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, length: int) -> bool:
        """
        Allocate blocks for a new sequence.

        Args:
            seq_id: Unique sequence identifier
            length: Initial sequence length

        Returns:
            bool: True if allocation successful
        """
        num_blocks_needed = (length + self.block_size - 1) // self.block_size

        blocks = self.block_allocator.allocate_blocks(num_blocks_needed)
        if not blocks:
            return False

        self.seq_blocks[seq_id] = blocks
        self.seq_lengths[seq_id] = length
        return True

    def append_tokens(self, seq_id: int, num_tokens: int) -> bool:
        """
        Extend sequence by num_tokens, allocating new blocks if needed.

        Returns:
            bool: True if extension successful
        """
        if seq_id not in self.seq_blocks:
            return False

        current_length = self.seq_lengths[seq_id]
        new_length = current_length + num_tokens

        # Check if we need more blocks
        current_blocks = self.seq_blocks[seq_id]
        blocks_needed = (new_length + self.block_size - 1) // self.block_size

        if blocks_needed > len(current_blocks):
            # Allocate additional blocks
            additional_needed = blocks_needed - len(current_blocks)
            new_blocks = self.block_allocator.allocate_blocks(additional_needed)
            if not new_blocks:
                return False
            current_blocks.extend(new_blocks)

        self.seq_lengths[seq_id] = new_length
        return True

    def get_block_table(self, seq_id: int) -> List[int]:
        """Get list of blocks for a sequence"""
        return self.seq_blocks.get(seq_id, [])

    def compute_paged_attention(
        self,
        query: torch.Tensor,  # [batch_size, num_heads, head_dim]
        seq_ids: List[int],
        context_lengths: List[int]
    ) -> torch.Tensor:
        """
        Compute attention using paged KV cache.

        This implements block-sparse attention where we only load
        the blocks that contain valid tokens for each sequence.
        """
        batch_size = query.size(0)
        num_heads = query.size(1)
        head_dim = query.size(2)

        output = torch.zeros_like(query)

        for batch_idx, seq_id in enumerate(seq_ids):
            if seq_id not in self.seq_blocks:
                continue

            blocks = self.seq_blocks[seq_id]
            context_len = context_lengths[batch_idx]

            # Gather K and V from blocks
            k_list = []
            v_list = []
            tokens_remaining = context_len

            for block_num in blocks:
                if tokens_remaining <= 0:
                    break

                k_block, v_block = self.block_allocator.get_kv_cache(block_num)
                # Only take valid tokens from this block
                tokens_in_block = min(self.block_size, tokens_remaining)

                if tokens_in_block == self.block_size:
                    k_list.append(k_block)
                    v_list.append(v_block)
                else:
                    k_list.append(k_block[:tokens_in_block])
                    v_list.append(v_block[:tokens_in_block])

                tokens_remaining -= tokens_in_block

            if k_list:
                k = torch.cat(k_list, dim=0)  # [context_len, num_heads, head_dim]
                v = torch.cat(v_list, dim=0)

                # Compute attention
                q = query[batch_idx].unsqueeze(0)  # [1, num_heads, head_dim]

                # [1, num_heads, head_dim] @ [context_len, num_heads, head_dim].T
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                # Causal mask is implicit since we only attend to previous tokens
                attn_weights = torch.softmax(scores, dim=-1)

                # [1, num_heads, context_len] @ [context_len, num_heads, head_dim]
                out = torch.matmul(attn_weights, v)
                output[batch_idx] = out.squeeze(0)

        return output

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks allocated to a sequence"""
        if seq_id not in self.seq_blocks:
            return

        for block_num in self.seq_blocks[seq_id]:
            self.block_allocator.free_block(block_num)

        del self.seq_blocks[seq_id]
        del self.seq_lengths[seq_id]

    def get_memory_stats(self) -> Dict[str, any]:
        """Get memory usage statistics"""
        return {
            'block_allocator': self.block_allocator.get_memory_usage(),
            'num_sequences': len(self.seq_blocks),
            'total_tokens': sum(self.seq_lengths.values())
        }