# Qwen Inference Engine - Claude Context

## Project Vision

自研推理引擎，目标解决 llama.cpp / vLLM 对 MoE 类模型（如 Qwen3.5-35B-A3B）跑不满显卡性能的问题。
最终目标：学术发表 + 商业化落地。

核心差异化：
- 自定义量化流程（非依赖第三方 GPTQ/AWQ 工具，自己做 calibration + 量化）
- 持久化 KV Cache（落盘到 HDD/SSD，支持跨会话复用）
- OpenAI 兼容 API（对接各类 API Gateway 无缝替换）
- 针对 MoE 稀疏激活模式的 GPU 调度优化

## Model Characteristics

Qwen3.5-35B-A3B-GPTQ-Int4 是 MoE（Mixture of Experts）架构：
- 总参数 35B，每次推理仅激活 ~3B（A3B = Active 3B）
- Expert 路由决定哪些参数被激活，导致显存访问模式高度不规则
- **transformers 库不支持 Qwen3.5 MoE 架构，官方无 Python 脚本，需自研实现**
- vLLM 在 4 卡 V100 SXM2 上存在 bug：GPU 利用率仅 100-130W（远低于 300W TDP 目标），推理速度慢
- 本引擎的核心优势：
  - 感知 expert 激活模式，做针对性的显存预取和计算调度
  - 优化 GPU 功耗利用，目标跑满 V100 SXM2 的 300W TDP
  - NVLink 拓扑感知调度（当前 8×V100 SXM2 双 NVLink 组，每组 4 卡）

## Code Conventions

- Python 3.12+，类型标注必须
- 异步用 asyncio（API 层），推理核心用同步 + torch.cuda.stream
- 依赖管理：requirements.txt（当前），后续迁移 pyproject.toml
- 日志统一用 logging module，不用 print
- GPU 相关代码必须有 OOM 保护（try/except torch.cuda.OutOfMemoryError）
- 配置统一走 Config dataclass + .env，不硬编码

## Architecture

### Current Core Components

1. **Paged Attention** (`qwen_infer/attention/`)
   - Block-based KV cache，参考 vLLM 但更严格的显存记账
   - 120,000+ 预分配 blocks，支持 CoW fork
   - 非连续存储，支持 >200k token 序列

2. **Memory Management** (`qwen_infer/memory/`)
   - `GPUMemoryManager`: 单 GPU 显存管理，水位线机制
   - `KVCacheManager`: 细粒度 KV cache 控制，LRU 驱逐
   - 安全余量 2GB/GPU，四级内存压力（Normal/Warning/Critical/Eviction）

3. **Tensor Parallelism** (`qwen_infer/engine/tensor_parallel.py`)
   - 2-GPU Column/Row parallel
   - QKV 投影用 ColumnParallel，输出投影用 RowParallel
   - SwiGLU MLP + RMSNorm 的完整 Transformer 层

4. **GPTQ Loader** (`qwen_infer/models/gptq_loader.py`)
   - 4-bit 量化权重加载，运行时反量化
   - 多 GPU 权重分发

5. **Inference Engine** (`qwen_infer/engine/inference_engine.py`)
   - 顶层编排：模型加载 → 显存管理 → Paged Attention → 生成
   - 序列生命周期管理

### Planned Components

6. **Custom Quantization Pipeline** (`qwen_infer/quantization/` — TODO)
   - 自研量化：calibration dataset 选取 + 量化策略
   - 支持 GPTQ / AWQ / 自定义混合精度
   - 目标：比通用工具更好地适配 MoE 稀疏结构

7. **Persistent KV Cache** (`qwen_infer/cache/` — TODO)
   - KV Cache 持久化到 HDD/SSD（数据库存储）
   - 三级锁粒度（悲观锁）：
     - **User 级**: 用户维度的 KV cache 隔离
     - **Task 级**: 同一用户的不同任务/对话互不干扰
     - **Session 级**: 单次推理会话内的 cache 一致性
   - 支持跨会话 cache 复用，减少重复 prefill
   - 冷热分层：GPU → CPU → Disk，按访问频率自动迁移

8. **OpenAI-Compatible API** (`qwen_infer/api/` — TODO)
   - 完整兼容 OpenAI Chat Completions API (`/v1/chat/completions`)
   - 兼容 `/v1/models`, `/v1/completions`, `/v1/embeddings`
   - SSE streaming 支持
   - 可直接对接 OpenRouter / LiteLLM / FastGPT 等 API Gateway
   - 认证、限流、usage 计量

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

- 120,000 blocks × 16 tokens/block = 1.92M token capacity
- Paged allocation 非连续存储
- LRU 驱逐 + 显存碎片整理

### Strict Memory Management

四级内存压力：
1. **Normal**: <70% utilization
2. **Warning**: 70-85%
3. **Critical**: >85% → 触发清理
4. **Eviction**: >95% → 紧急驱逐

### Multi-GPU Strategy

当前硬件配置：8× V100 16GB，双 NVLink 组（每组 4 卡）
- NVLink Group 0: GPUs 0,1,2,3（组内 NVLink 互联）
- NVLink Group 1: GPUs 4,5,6,7（组内 NVLink 互联）
- 组间通过 PCIe 通信（带宽较低）

调度策略：
- 优先在同一 NVLink 组内做张量并行（利用高带宽 300GB/s）
- 跨组通信最小化（PCIe 仅 ~16GB/s）
- Expert 路由感知：将频繁协作的 expert 放在同一 NVLink 组
- 默认测试用 GPUs 2,3（同组），生产环境可扩展到 4 卡或 8 卡

未来架构支持：
- Ampere (A100): NVLink 3.0 (600GB/s)，NVSwitch 全连接
- Hopper (H100): NVLink 4.0 (900GB/s)，Transformer Engine 支持
- 需要针对不同架构的 NVLink 拓扑做自适应调度

### Persistent KV Cache Design (TODO)

存储模型：
```
User (排他锁 — 写时锁定整个用户的 cache 空间)
 └── Task (排他锁 — 同一用户不同对话/任务隔离)
      └── Session (读写锁 — 推理中持写锁，cache 复用时持读锁)
           └── KV Cache Blocks → DB (HDD/SSD)
```

锁策略细节：
- 全部悲观锁，不做乐观重试（推理场景下冲突代价太高）
- 锁获取顺序固定为 User → Task → Session，防止死锁
- 超时机制：锁等待超过阈值则放弃并返回错误，不无限阻塞
- Session 层用读写锁：多个请求可并发读取已有 cache，写入（生成新 token）时排他

关键设计决策：
- 冷数据异步落盘，热数据常驻 GPU
- 序列化格式：按 block 粒度存储，header 记录 shape/dtype/GPU 来源，支持跨 GPU 恢复
- DB 选型倾向 RocksDB（顺序写性能好，适合 append-heavy 的 KV cache 场景）
- 备选：mmap 直接映射文件（更低延迟但需要自己管理并发）

### API Compatibility (TODO)

必须兼容的接口：
- `POST /v1/chat/completions` (核心)
- `POST /v1/completions`
- `GET /v1/models`
- SSE streaming (`stream: true`)
- `usage` 字段（prompt_tokens, completion_tokens, total_tokens）

## Known Bugs

- `kv_cache_manager.py:167` — `extend_sequence` 中 `seq_blocks` 缺少 `self.` 前缀
- `paged_attention.py:73` — `torch.cuda.Semaphore()` 不存在于 PyTorch API
- `memory_manager.py:161` — `num_elements` 计算的位运算优先级有误
- `inference_engine.py` — `generate()` 目前只返回 dummy token，实际推理未接入

## Testing

```bash
python tests/test_engine.py
```

## Benchmarking

```bash
python benchmarks/benchmark.py
```

## Performance Targets (论文对比基线)

| Metric | vLLM baseline | llama.cpp baseline | 本引擎目标 |
|--------|--------------|-------------------|-----------|
| Throughput (tok/s, batch=1) | ~40 | ~30 | >60 |
| Throughput (tok/s, batch=8) | ~200 | N/A | >350 |
| GPU utilization (power) | 100-130W | ~150W | >240W (接近 250W TDP) |
| GPU utilization (compute) | ~50-60% | ~40-50% | >80% |
| TTFT (Time to First Token, 4k prompt) | ~800ms | ~1200ms | <500ms |
| 200k context prefill | OOM/slow | 不支持 | <60s |
| KV cache 跨会话恢复 | 不支持 | 不支持 | <2s (hot) / <10s (cold) |
| NVLink 带宽利用率 | <30% | N/A | >70% |

测试环境：8× V100 16GB (双 NVLink 组)，Qwen3.5-35B-A3B-GPTQ-Int4

vLLM 已知问题：
- 4 卡 V100 上 GPU 功耗仅 100-130W（正常应接近 250W TDP）
- MoE expert 调度未优化，导致 GPU 大量空闲等待
- 跨 NVLink 组通信未优化，PCIe 成为瓶颈

## Roadmap

### Phase 1: Core Engine (Current)
- [x] Paged Attention 基础实现
- [x] 2-GPU Tensor Parallelism
- [x] GPTQ 4-bit 加载
- [x] 显存管理框架
- [ ] 修复已知 bugs
- [ ] 接入实际推理（generate 不再返回 dummy）

### Phase 2: Custom Quantization (可与 Phase 3 并行)
- [ ] 自研量化 pipeline（calibration + quantize）
- [ ] MoE expert 级别的混合精度（热门 expert 用更高精度）
- [ ] 量化精度 vs 性能的 benchmark 对比

### Phase 3: Persistent KV Cache (可与 Phase 2 并行)
- [ ] RocksDB 存储层实现
- [ ] User/Task/Session 三级锁（固定获取顺序 + 超时）
- [ ] GPU ↔ CPU ↔ Disk 冷热迁移
- [ ] 跨会话 cache 复用 + 一致性校验

### Phase 4: API & Production
- [ ] OpenAI 兼容 API 层
- [ ] SSE streaming
- [ ] 认证 + 限流 + usage 计量
- [ ] API Gateway 对接测试

### Phase 5: Optimization & Paper
- [ ] Custom CUDA kernels (paged attention + MoE dispatch)
- [ ] Speculative decoding（利用 MoE 小 expert 做 draft）
- [ ] Continuous batching
- [ ] Dynamic batch size
- [ ] Prefix caching
- [ ] 性能对比实验（vs vLLM, llama.cpp, TGI, SGLang）
- [ ] 论文撰写（切入点：MoE 感知调度 + 持久化 KV cache 的联合优化）

## Reference

- vLLM: https://arxiv.org/abs/2309.06180
- PagedAttention: https://arxiv.org/abs/2309.06180
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- SGLang: https://arxiv.org/abs/2312.07104
- Qwen3.5: https://huggingface.co/Qwen
- OpenAI API spec: https://platform.openai.com/docs/api-reference
- RocksDB: https://rocksdb.org/
