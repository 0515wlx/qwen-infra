"""
Tensor Parallelism for Multi-GPU Inference

Implements 2-GPU tensor parallelism for Qwen models.
Splits model weights and computation across GPUs 2 and 3.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class TensorParallelGroup:
    """Manages tensor parallelism group for 2-GPU setup"""

    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.world_size = len(gpu_ids)
        self.rank = None
        self.group = None

    def initialize(self):
        """Initialize process group for tensor parallelism"""
        if not dist.is_initialized():
            # Set single-process multi-GPU mode
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=0  # Single process controls all GPUs
            )

        # Create local group
        ranks = list(range(self.world_size))
        self.group = dist.new_group(ranks)
        self.rank = 0

        # Ensure all GPUs are ready
        for gpu_id in self.gpu_ids:
            torch.cuda.set_device(gpu_id)
            torch.cuda.synchronize()

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    Splits output features across GPUs.
    Used for Q, K, V projections in attention.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: TensorParallelGroup,
        bias: bool = True,
        gather_output: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.gather_output = gather_output

        # Split output features
        assert out_features % tp_group.world_size == 0
        self.output_size_per_partition = out_features // tp_group.world_size

        # Create linear layer
        self.linear = nn.Linear(
            in_features,
            self.output_size_per_partition,
            bias=bias
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Simple linear on local partition
        output = self.linear(input)

        if self.gather_output and self.tp_group.world_size > 1:
            # Gather outputs from all GPUs
            outputs = [torch.empty_like(output) for _ in range(self.tp_group.world_size)]
            dist.all_gather(outputs, output, group=self.tp_group.group)
            output = torch.cat(outputs, dim=-1)

        return output

class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    Splits input features across GPUs.
    Used for output projections.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: TensorParallelGroup,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group

        # Split input features
        assert in_features % tp_group.world_size == 0
        self.input_size_per_partition = in_features // tp_group.world_size

        # Create linear layer
        self.linear = nn.Linear(
            self.input_size_per_partition,
            out_features,
            bias=bias
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If input is already split, just apply linear
        # Otherwise, split manually
        if input.size(-1) != self.input_size_per_partition:
            # Split along last dimension
            splits = torch.chunk(input, self.tp_group.world_size, dim=-1)
            input = splits[self.tp_group.rank]

        output = self.linear(input)

        # All-reduce across GPUs
        if self.tp_group.world_size > 1:
            dist.all_reduce(output, group=self.tp_group.group)

        return output

class ParallelAttention(nn.Module):
    """Multi-head attention with tensor parallelism"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int],
        tp_group: TensorParallelGroup,
        head_dim: int = 128
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        self.tp_group = tp_group

        # Total sizes
        self.q_size = num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        # Split across GPUs
        world_size = tp_group.world_size
        assert num_heads % world_size == 0
        self.num_heads_per_partition = num_heads // world_size
        self.num_kv_heads_per_partition = self.num_kv_heads // world_size

        # Q, K, V projections - column parallel
        self.q_proj = ColumnParallelLinear(
            hidden_size, self.q_size, tp_group,
            gather_output=False
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, self.kv_size, tp_group,
            gather_output=False
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, self.kv_size, tp_group,
            gather_output=False
        )

        # Output projection - row parallel
        self.o_proj = RowParallelLinear(
            self.q_size, hidden_size, tp_group
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()

        # Q, K, V projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)

        # Update KV cache if provided
        if kv_cache_k is not None:
            k = torch.cat([kv_cache_k, k], dim=1)
        if kv_cache_v is not None:
            v = torch.cat([kv_cache_v, v], dim=1)

        # Compute attention (simplified)
        kv_len = k.size(1)
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        if seq_len > 1:
            mask = torch.triu(
                torch.ones(seq_len, kv_len, device=scores.device),
                diagonal=kv_len - seq_len + 1
            ).bool()
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(out)

        return output, k, v

class ParallelMLP(nn.Module):
    """MLP with tensor parallelism"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_group: TensorParallelGroup
    ):
        super().__init__()
        self.tp_group = tp_group

        # Gate and up projections - column parallel
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, tp_group
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, tp_group
        )

        # Down projection - row parallel
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, tp_group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        return x

class ParallelTransformerLayer(nn.Module):
    """Transformer layer with tensor parallelism"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        tp_group: TensorParallelGroup,
        num_kv_heads: Optional[int] = None
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.self_attn = ParallelAttention(
            hidden_size, num_heads, num_kv_heads, tp_group
        )
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.mlp = ParallelMLP(hidden_size, intermediate_size, tp_group)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, k, v = self.self_attn(hidden_states, kv_cache[0] if kv_cache else None, kv_cache[1] if kv_cache else None)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, (k, v)

class TensorParallelModel(nn.Module):
    """Model wrapper for tensor parallel inference"""

    def __init__(
        self,
        config: Dict,
        tp_group: TensorParallelGroup
    ):
        super().__init__()
        self.tp_group = tp_group
        self.config = config

        # Extract config
        self.hidden_size = config.get('hidden_size', 3584)
        self.num_heads = config.get('num_attention_heads', 28)
        self.num_layers = config.get('num_hidden_layers', 60)
        self.intermediate_size = config.get('intermediate_size', 18944)
        self.num_kv_heads = config.get('num_key_value_heads', 4)
        self.vocab_size = config.get('vocab_size', 152064)

        # Embedding (replicated on all GPUs)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            ParallelTransformerLayer(
                self.hidden_size,
                self.num_heads,
                self.intermediate_size,
                self.tp_group,
                self.num_kv_heads
            )
            for _ in range(self.num_layers)
        ])

        self.norm = nn.RMSNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Transformer layers
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches else None
            hidden_states, new_kv = layer(hidden_states, kv_cache)
            new_kv_caches.append(new_kv)

        # Final norm and head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, new_kv_caches