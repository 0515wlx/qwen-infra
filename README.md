# Qwen Inference Engine

A high-performance inference engine for Qwen series models, specifically optimized for Qwen3.5-35B-A3B-GPTQ-Int4.

## Features

- **Paged Attention**: Efficient KV cache management with block-based allocation
- **Strict Memory Management**: Fine-grained GPU memory control with watermarks and safety margins
- **Multi-GPU Support**: Tensor parallelism across 2 GPUs (tested on GPUs 2 & 3)
- **Long Sequence Support**: Optimized for sequences >200k tokens
- **GPTQ 4-bit**: Native support for GPTQ quantized models

## Architecture

```
qwen_infer/
├── attention/          # Paged attention implementation
│   └── paged_attention.py
├── engine/            # Core inference engine
│   ├── inference_engine.py
│   └── tensor_parallel.py
├── memory/            # Memory management
│   ├── memory_manager.py
│   └── kv_cache_manager.py
├── models/            # Model loading
│   └── gptq_loader.py
├── config/            # Configuration
│   └── settings.py
└── utils/             # Utilities
    └── memory_utils.py
```

## Quick Start

1. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your paths and GPU settings
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests**:
   ```bash
   python tests/test_engine.py
   ```

4. **Basic usage**:
   ```python
   from qwen_infer import InferenceEngine, Config

   config = Config()
   engine = InferenceEngine(config)
   engine.initialize()

   # Create sequence and generate
   seq_id = engine.create_sequence(prompt_tokens)
   for token in engine.generate(seq_id, max_new_tokens=100):
       print(token)
   ```

## Configuration

All settings are managed via `.env` file:

```bash
# Model path
MODEL_PATH=/mnt/disk/models/Qwen3.5-35B-A3B-GPTQ-Int4

# GPUs to use (0,1 or 2,3)
CUDA_VISIBLE_DEVICES=2,3

# Memory settings
MAX_GPU_MEMORY_GB=48
SAFETY_MARGIN_GB=2.0

# KV cache
BLOCK_SIZE=16
NUM_GPU_BLOCKS=120000

# Sequence limits
MAX_SEQUENCE_LENGTH=250000
```

## Testing

Test suite validates:
- Model loading on GPUs 2 and 3
- Memory management with safety margins
- Paged attention functionality
- Long sequence support (>200k tokens)
- Multi-GPU distribution

Run with:
```bash
python tests/test_engine.py
```

## Performance Tuning

### Memory Management
- `SAFETY_MARGIN_GB`: Reserved memory buffer per GPU
- `NUM_GPU_BLOCKS`: Pre-allocated KV cache blocks
- `BLOCK_SIZE`: Tokens per block (default: 16)

### Multi-GPU
- `TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism
- Automatic load balancing across GPUs

## License

Apache 2.0