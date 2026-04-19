# Qwen Inference Engine - Claude Context

## Project Overview

This is a custom inference engine for the Qwen series of LLMs, specifically optimized for:
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- GPUs: 2-GPU tensor parallelism (GPUs 2,3 for testing)
- Sequence Length: >200k tokens
- Quantization: GPTQ 4-bit

## Architecture

### Core Components

1. **Paged Attention** (`qwen_infer/attention/`)
   - Block-based KV cache management
   - Similar to vLLM's approach but with stricter memory accounting
   - Supports 120,000+ pre-allocated blocks

2. **Memory Management** (`qwen_infer/memory/`)
   - `GPUMemoryManager`: Per-GPU memory with watermarks
   - `KVCacheManager`: Fine-grained KV cache control
   - Safety margins (2GB per GPU default)
   - Memory pressure detection and emergency cleanup

3. **Tensor Parallelism** (`qwen_infer/engine/tensor_parallel.py`)
   - 2-GPU column/row parallel linear layers
   - Distributed attention and MLP computation
   - Automatic load balancing

4. **GPTQ Loader** (`qwen_infer/models/gptq_loader.py`)
   - Loads 4-bit quantized weights
   - On-the-fly dequantization during inference
   - Multi-GPU weight distribution

### Key Configuration

Environment variables in `.env`:
```bash
MODEL_PATH=/mnt/disk/models/Qwen3.5-35B-A3B-GPTQ-Int4
CUDA_VISIBLE_DEVICES=2,3
TENSOR_PARALLEL_SIZE=2
MAX_GPU_MEMORY_GB=48
SAFETY_MARGIN_GB=2.0
NUM_GPU_BLOCKS=120000
MAX_SEQUENCE_LENGTH=250000
```

## Critical Implementation Details

### Long Sequence Support (>200k)

The engine uses aggressive pre-allocation:
- 120,000 blocks × 16 tokens/block = 1.92M token capacity
- Paged allocation allows non-contiguous storage
- LRU eviction for inactive sequences
- Memory defragmentation on pressure

### Strict Memory Management

Three-level memory pressure system:
1. **Normal**: <70% utilization
2. **Warning**: 70-85% utilization
3. **Critical**: >85% utilization → triggers cleanup
4. **Eviction**: >95% utilization → emergency measures

### Multi-GPU Strategy

- GPUs 2,3 are default (0,1 may be occupied in test env)
- Each GPU maintains its own KV cache pool
- Column parallelism for Q,K,V projections
- Row parallelism for output projections

## Testing

Run test suite:
```bash
python tests/test_engine.py
```

Key test scenarios:
- Basic initialization on GPUs 2,3
- Memory safety margin enforcement
- Paged attention allocation/deallocation
- 200k+ token sequence creation
- Multi-GPU weight distribution

## Benchmarking

```bash
python benchmarks/benchmark.py
```

Measures:
- Memory scalability (1k to 200k tokens)
- Generation throughput (tok/s)
- Latency per token
- Long sequence handling

## Future Improvements

- [ ] CUDA kernel optimization for paged attention
- [ ] Speculative decoding
- [ ] Continuous batching
- [ ] Prefix caching
- [ ] Dynamic batch size adjustment

## Reference

- vLLM paper: https://arxiv.org/abs/2309.06180
- GPTQ: https://arxiv.org/abs/2210.17323
- Qwen3.5: https://huggingface.co/Qwen