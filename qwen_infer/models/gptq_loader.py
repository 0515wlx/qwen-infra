"""
GPTQ Model Loader

Loads and manages GPTQ 4-bit quantized models.
Optimized for Qwen3.5-35B-A3B-GPTQ-Int4.
"""

import torch
import torch.nn as nn
import json
import os
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class GPTQConfig:
    """GPTQ quantization configuration"""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        static_groups: bool = False,
        sym: bool = True,
        true_sequential: bool = True
    ):
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.static_groups = static_groups
        self.sym = sym
        self.true_sequential = true_sequential

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GPTQConfig':
        return cls(
            bits=config_dict.get('bits', 4),
            group_size=config_dict.get('group_size', 128),
            desc_act=config_dict.get('desc_act', False),
            static_groups=config_dict.get('static_groups', False),
            sym=config_dict.get('sym', True),
            true_sequential=config_dict.get('true_sequential', True)
        )

class QuantizedLinear(nn.Module):
    """
    Linear layer with GPTQ 4-bit quantization.

    Dequantizes on-the-fly during forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_config: GPTQConfig,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = quant_config

        # GPTQ uses 4-bit quantization
        self.bits = quant_config.bits
        self.group_size = quant_config.group_size

        # Packed weights (4-bit packed into 8-bit integers)
        # Each byte stores 2 4-bit weights
        packed_size = (in_features * out_features) // 2
        self.qweight = nn.Parameter(
            torch.empty(packed_size, dtype=torch.uint8, device='cuda'),
            requires_grad=False
        )

        # Quantization parameters per group
        num_groups = in_features // quant_config.group_size
        self.qzeros = nn.Parameter(
            torch.empty((out_features, num_groups), dtype=torch.int32, device='cuda'),
            requires_grad=False
        )
        self.scales = nn.Parameter(
            torch.empty((out_features, num_groups), dtype=torch.float16, device='cuda'),
            requires_grad=False
        )

        self.register_buffer(
            'g_idx',
            torch.empty(in_features, dtype=torch.int32, device='cuda')
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16, device='cuda'))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly
        # This is the main computational overhead of GPTQ

        # Unpack 4-bit weights
        weight = self.unpack_weights()

        # Apply quantization parameters
        num_groups = self.in_features // self.group_size
        for g in range(num_groups):
            start = g * self.group_size
            end = start + self.group_size
            weight[:, start:end] = (
                weight[:, start:end].float() - self.qzeros[:, g:g+1]
            ) * self.scales[:, g:g+1]

        # Cast to input dtype and compute
        weight = weight.to(x.dtype)
        output = torch.nn.functional.linear(x, weight, self.bias)

        return output

    def unpack_weights(self) -> torch.Tensor:
        """
        Unpack 4-bit weights from packed uint8 storage.

        Returns: [out_features, in_features] float tensor
        """
        # Unpack 4-bit weights
        # qweight stores pairs of 4-bit values in each byte
        # Lower 4 bits: weight[i], Upper 4 bits: weight[i+1]

        # Expand packed weights
        weight_byte = self.qweight.to(torch.int32)
        low = weight_byte & 0xF  # Lower 4 bits
        high = (weight_byte >> 4) & 0xF  # Upper 4 bits

        # Interleave to get full weight matrix
        weight = torch.stack([low, high], dim=1).view(-1)
        weight = weight[:self.out_features * self.in_features]
        weight = weight.view(self.out_features, self.in_features)

        return weight.float()

    @classmethod
    def from_file(
        cls,
        in_features: int,
        out_features: int,
        quant_config: GPTQConfig,
        checkpoint: Dict[str, torch.Tensor],
        prefix: str,
        bias: bool = False
    ) -> 'QuantizedLinear':
        """Load from checkpoint file"""
        layer = cls(in_features, out_features, quant_config, bias)

        layer.qweight.data = checkpoint[f'{prefix}.qweight']
        layer.qzeros.data = checkpoint[f'{prefix}.qzeros']
        layer.scales.data = checkpoint[f'{prefix}.scales']

        if f'{prefix}.g_idx' in checkpoint:
            layer.g_idx.data = checkpoint[f'{prefix}.g_idx']

        if bias and f'{prefix}.bias' in checkpoint:
            layer.bias.data = checkpoint[f'{prefix}.bias']

        return layer

class GPTQModelLoader:
    """
    Loads GPTQ quantized models.

    Handles:
    - Model config loading
    - Weight loading and dequantization
    - Multi-GPU distribution
    """

    def __init__(self, model_path: str, quant_config: Optional[GPTQConfig] = None):
        self.model_path = Path(model_path)
        self.quant_config = quant_config
        self.model_config = None

    def load_config(self) -> Dict:
        """Load model configuration"""
        config_path = self.model_path / 'config.json'

        with open(config_path, 'r') as f:
            self.model_config = json.load(f)

        # Load GPTQ config if not provided
        if self.quant_config is None:
            quant_config_path = self.model_path / 'quant_config.json'
            if quant_config_path.exists():
                with open(quant_config_path, 'r') as f:
                    qc = json.load(f)
                self.quant_config = GPTQConfig.from_dict(qc)
            else:
                # Default for 4-bit GPTQ
                self.quant_config = GPTQConfig(bits=4, group_size=128)

        return self.model_config

    def list_checkpoint_files(self) -> list:
        """List all checkpoint files in the model directory"""
        patterns = ['*.safetensors', '*.bin', '*.pt']
        files = []
        for pattern in patterns:
            files.extend(self.model_path.glob(pattern))
        return sorted(files)

    def load_checkpoint(self, file_path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Load a checkpoint file"""
        path = Path(file_path)

        if path.suffix == '.safetensors':
            try:
                from safetensors.torch import load_file
                return load_file(str(path), device=device)
            except ImportError:
                logger.warning("safetensors not installed, falling back to torch.load")
                return torch.load(path, map_location=device)
        else:
            return torch.load(path, map_location=device, weights_only=True)

    def load_model_for_gpus(
        self,
        gpu_ids: list[int],
        device_map: Optional[Dict[str, int]] = None
    ) -> nn.Module:
        """
        Load model distributed across multiple GPUs.

        Args:
            gpu_ids: List of GPU IDs to use
            device_map: Optional explicit device mapping for layers

        Returns:
            nn.Module: Loaded model
        """
        if self.model_config is None:
            self.load_config()

        logger.info(f"Loading GPTQ model from {self.model_path}")
        logger.info(f"Quantization: {self.quant_config.bits}-bit, group_size={self.quant_config.group_size}")
        logger.info(f"Target GPUs: {gpu_ids}")

        # Get checkpoint files
        checkpoint_files = self.list_checkpoint_files()
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {self.model_path}")

        logger.info(f"Found {len(checkpoint_files)} checkpoint files")

        # Load all weights into memory (may need to optimize for very large models)
        all_weights = {}
        for ckpt_file in checkpoint_files:
            logger.info(f"Loading {ckpt_file.name}...")
            weights = self.load_checkpoint(ckpt_file, device='cpu')
            all_weights.update(weights)

        logger.info(f"Total parameters loaded: {len(all_weights)}")

        # TODO: Build and distribute model
        # This is a placeholder - actual implementation would construct
        # the model architecture and distribute weights across GPUs

        return all_weights  # Return weights dict for now

    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage for the model"""
        if self.model_config is None:
            self.load_config()

        # Model size in billion parameters (35B for Qwen3.5-35B)
        num_params = self.model_config.get('num_parameters', 35e9)

        # GPTQ 4-bit quantization: ~0.5 bytes per parameter
        bytes_per_param = self.quant_config.bits / 8
        model_size_gb = (num_params * bytes_per_param) / (1024**3)

        # KV cache for max sequence length
        max_seq_len = self.model_config.get('max_position_embeddings', 250000)
        num_layers = self.model_config.get('num_hidden_layers', 60)
        num_kv_heads = self.model_config.get('num_key_value_heads', 4)
        head_dim = self.model_config.get('head_dim', 128)
        batch_size = 4  # Default max batch size

        # KV cache size
        kv_cache_gb = (
            batch_size * max_seq_len * num_layers * num_kv_heads * head_dim * 2 * 2
        ) / (1024**3)  # *2 for K+V, *2 for fp16

        return {
            'model_size_gb': model_size_gb,
            'kv_cache_max_gb': kv_cache_gb,
            'total_required_gb': model_size_gb + kv_cache_gb,
            'per_gpu_gb': (model_size_gb + kv_cache_gb) / 2  # Assuming 2 GPUs
        }