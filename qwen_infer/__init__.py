"""
Qwen Inference Engine

A high-performance inference engine for Qwen series models with:
- Paged Attention mechanism
- Strict GPU memory management
- Multi-GPU tensor parallelism
- Long sequence support (>200k tokens)
"""

__version__ = "0.1.0"
__author__ = "Qwen Inference Team"

from qwen_infer.engine.inference_engine import InferenceEngine
from qwen_infer.config.settings import Config

__all__ = ['InferenceEngine', 'Config']