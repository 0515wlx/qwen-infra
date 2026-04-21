# Qwen Inference Engine - Implementation Plan

本文档详细列出每个开发阶段的具体任务和实现细节。

---

## Phase 1: Core Engine Stabilization (当前阶段)

**目标**: 修复已知 bugs，完成基础推理流程，建立可运行的 baseline

### 1.1 Bug 修复

**任务**: 修复 4 个已知 bugs
- [ ] `kv_cache_manager.py:167` — `extend_sequence` 中 `seq_blocks` 加 `self.` 前缀
- [ ] `paged_attention.py:73` — 移除 `torch.cuda.Semaphore()`，改用 `threading.Lock` 或去掉（PyTorch 无此 API）
- [ ] `memory_manager.py:161` — 修正 `num_elements` 计算的位运算优先级
- [ ] `inference_engine.py` — 实现真实的 `generate()` 逻辑，替换 dummy token

**验证**: 运行 `python tests/test_engine.py`，所有测试通过

### 1.2 Qwen3.5 MoE 模型实现

**任务**: 自研 Qwen3.5 MoE 架构（transformers 不支持）

**子任务**:
1. **模型配置解析** (`qwen_infer/models/qwen_config.py`)
   - 从 `config.json` 读取 MoE 参数：
     - `num_experts`: 总 expert 数量
     - `num_experts_per_tok`: 每 token 激活的 expert 数（通常 2）
     - `expert_capacity`: expert 容量限制
   - 解析 attention/MLP 层配置

2. **Expert 路由层** (`qwen_infer/models/moe_router.py`)
   - 实现 Top-K 路由：`router_logits = Linear(hidden_states)` → `topk(k=num_experts_per_tok)`
   - 计算 expert 权重（softmax over top-k）
   - 返回 `(expert_ids, expert_weights)` 用于后续调度

3. **MoE FFN 层** (`qwen_infer/models/moe_ffn.py`)
   - 每个 expert 是独立的 FFN（SwiGLU 结构）
   - 根据路由结果，只加载激活的 expert 权重到 GPU
   - 实现 expert 并行计算 + 加权求和

4. **完整 Transformer 层** (`qwen_infer/models/qwen_transformer.py`)
   - Attention (标准 GQA) + RMSNorm
   - MoE FFN + RMSNorm
   - Residual connections

5. **模型加载器** (`qwen_infer/models/qwen_loader.py`)
   - 从 safetensors/bin 文件加载权重
   - 支持 GPTQ 4-bit 量化权重
   - 多 GPU 权重分发（张量并行）

**验证**:
- 单层 forward pass 输出 shape 正确
- 对比官方 Qwen 输出（如果有参考实现）

### 1.3 实际推理流程接入

**任务**: 将 MoE 模型接入 `InferenceEngine.generate()`

**子任务**:
1. **Prefill 阶段**
   - 输入 prompt tokens → embedding
   - 逐层 forward，KV cache 写入 paged blocks
   - 输出最后一个 token 的 logits

2. **Decode 阶段**
   - 循环生成：sample token → embedding → forward (复用 KV cache) → 下一个 token
   - 每步更新 paged attention 的 block 分配
   - 实现 sampling 策略（top-p, top-k, temperature）

3. **停止条件**
   - EOS token 或达到 max_length

**验证**:
- 输入简单 prompt，生成连贯文本
- 检查 KV cache 是否正确复用（decode 阶段不应重新计算 prefill）

### 1.4 NVLink 拓扑感知

**任务**: 检测 NVLink 拓扑，优化跨 GPU 通信

**子任务**:
1. **拓扑检测** (`qwen_infer/utils/nvlink_topology.py`)
   - 使用 `nvidia-smi topo -m` 或 `pynvml` 检测 NVLink 连接
   - 构建邻接矩阵：`nvlink_matrix[i][j] = bandwidth`
   - 识别 NVLink 组（强连通分量）

2. **调度策略**
   - 张量并行优先在同一 NVLink 组内（GPUs 0-3 或 4-7）
   - Expert 放置：频繁协作的 expert 放同组
   - 跨组通信用 async copy 隐藏延迟

**验证**:
- 打印检测到的 NVLink 拓扑
- 对比同组 vs 跨组通信延迟（应有 10x+ 差异）

---

## Phase 2: Custom Quantization Pipeline

**目标**: 自研量化工具，针对 MoE 稀疏结构优化

### 2.1 Calibration Dataset 准备

**任务**: 选取代表性数据集用于量化校准

**子任务**:
1. **数据集选择**
   - 通用：C4, WikiText, RedPajama 采样 512 条
   - 领域特定：根据应用场景（代码/对话/长文本）定制

2. **数据预处理** (`qwen_infer/quantization/calibration.py`)
   - Tokenize → 截断到 2048 tokens
   - 构建 DataLoader

**输出**: `calibration_data.pt`

### 2.2 激活值统计

**任务**: 收集模型各层的激活值分布

**子任务**:
1. **Hook 注册** (`qwen_infer/quantization/activation_observer.py`)
   - 在每层 Linear/MoE 前后插入 hook
   - 记录 min/max/histogram

2. **统计收集**
   - Forward calibration dataset
   - 保存统计信息到 `activation_stats.json`

**关键指标**:
- 每层的 dynamic range
- Expert 激活频率（热门 expert vs 冷门 expert）

### 2.3 量化策略设计

**任务**: 基于统计信息设计量化方案

**策略**:
1. **Per-channel quantization** (基础)
   - 每个输出 channel 独立 scale/zero_point

2. **MoE 混合精度** (核心创新)
   - 热门 expert (top 20% 激活频率) → INT8 或 FP16
   - 冷门 expert → INT4
   - 理由：热门 expert 对精度敏感，冷门 expert 可激进量化

3. **Group-wise quantization**
   - 权重按 128 元素分组量化（类似 GPTQ）

**实现**: `qwen_infer/quantization/quantizer.py`

### 2.4 量化执行

**任务**: 将 FP16 模型量化为 INT4/INT8

**子任务**:
1. **权重量化**
   ```python
   scale = (max - min) / (2^bits - 1)
   zero_point = -min / scale
   quantized = round(weight / scale + zero_point).clamp(0, 2^bits-1)
   ```

2. **保存量化模型**
   - 格式：safetensors，包含 `weight_int4`, `scale`, `zero_point`
   - 元数据：量化配置（bits, group_size, expert_precision_map）

**输出**: `Qwen3.5-35B-A3B-Custom-Int4/`

### 2.5 精度验证

**任务**: 对比量化前后的精度损失

**指标**:
- Perplexity (WikiText-2)
- MMLU / C-Eval 准确率
- 生成质量（人工评估）

**Benchmark**: `benchmarks/quantization_eval.py`

---

## Phase 3: Persistent KV Cache

**目标**: 实现跨会话 KV cache 复用，减少重复 prefill

### 3.1 存储层选型与实现

**任务**: 选择并实现 DB 存储

**方案对比**:
| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| RocksDB | 顺序写快，LSM-tree 适合 append | C++ 依赖，Python 绑定复杂 | ✅ 推荐 |
| SQLite | 轻量，零配置 | 写入性能差，不适合高并发 | 备选 |
| mmap | 最低延迟，直接内存映射 | 需自己管理并发和持久化 | 高级优化 |

**实现**: `qwen_infer/cache/storage_backend.py`

**接口**:
```python
class StorageBackend:
    def put(self, key: str, value: bytes) -> None
    def get(self, key: str) -> Optional[bytes]
    def delete(self, key: str) -> None
    def exists(self, key: str) -> bool
```

### 3.2 三级锁机制

**任务**: 实现 User → Task → Session 悲观锁

**实现**: `qwen_infer/cache/lock_manager.py`

**锁类型**:
1. **User 级排他锁**
   - 写入时锁定整个用户的 cache 空间
   - 防止同一用户的并发写冲突

2. **Task 级排他锁**
   - 不同对话/任务隔离
   - 同一用户可以并发访问不同 task

3. **Session 级读写锁**
   - 读锁：多个请求可并发读取已有 cache
   - 写锁：生成新 token 时排他

**死锁预防**:
- 固定获取顺序：User → Task → Session
- 超时机制：等待超过 30s 则放弃并返回错误

**实现**:
```python
class LockManager:
    def acquire_user_lock(self, user_id: str, timeout: float) -> bool
    def acquire_task_lock(self, user_id: str, task_id: str, timeout: float) -> bool
    def acquire_session_lock(self, user_id: str, task_id: str, session_id: str,
                            mode: Literal['read', 'write'], timeout: float) -> bool
    def release_all(self, user_id: str, task_id: str, session_id: str) -> None
```

### 3.3 KV Cache 序列化

**任务**: 将 GPU tensor 序列化到磁盘

**格式设计**:
```
Header (64 bytes):
  - magic: b'QWKV' (4 bytes)
  - version: uint16
  - block_id: uint32
  - shape: (seq_len, num_heads, head_dim) - 3x uint32
  - dtype: uint8 (0=fp16, 1=fp32, 2=bf16)
  - gpu_id: uint8
  - checksum: uint32 (CRC32)
  - reserved: 32 bytes

Data:
  - K tensor: shape[0] * shape[1] * shape[2] * dtype_size bytes
  - V tensor: same size
```

**实现**: `qwen_infer/cache/serializer.py`

**关键函数**:
```python
def serialize_kv_block(k: torch.Tensor, v: torch.Tensor, block_id: int) -> bytes
def deserialize_kv_block(data: bytes) -> Tuple[torch.Tensor, torch.Tensor, int]
```

### 3.4 冷热分层迁移

**任务**: GPU ↔ CPU ↔ Disk 自动迁移

**策略**:
1. **热数据** (最近 5 分钟访问) → GPU
2. **温数据** (5-30 分钟) → CPU 内存
3. **冷数据** (>30 分钟) → Disk

**实现**: `qwen_infer/cache/tier_manager.py`

**后台线程**:
- 每 60s 扫描一次访问时间戳
- 异步迁移（不阻塞推理）
- LRU 驱逐：GPU 显存不足时优先驱逐最久未访问的

### 3.5 跨会话复用

**任务**: 检测 cache 命中，跳过 prefill

**流程**:
1. 新请求到达 → 计算 prompt hash
2. 查询 DB：`get_cache(user_id, task_id, prompt_hash)`
3. 命中 → 加载 KV cache 到 GPU → 直接进入 decode
4. 未命中 → 正常 prefill → 保存 cache

**一致性校验**:
- 保存 prompt tokens 的 hash (SHA256)
- 加载时验证 hash 匹配

**实现**: `qwen_infer/cache/cache_manager.py`

---

## Phase 4: OpenAI-Compatible API

**目标**: 提供标准 API 接口，对接各类 Gateway

### 4.1 API 框架搭建

**技术栈**: FastAPI + uvicorn

**实现**: `qwen_infer/api/server.py`

**基础结构**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Qwen Inference API", version="1.0.0")

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "qwen3.5-35b-a3b", ...}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 调用 InferenceEngine
    pass
```

### 4.2 核心接口实现

**任务**: 实现 OpenAI 兼容的 4 个核心接口

#### 4.2.1 `/v1/chat/completions`

**Request**:
```json
{
  "model": "qwen3.5-35b-a3b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

**Response**:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen3.5-35b-a3b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

**实现**: `qwen_infer/api/chat.py`

#### 4.2.2 `/v1/completions`

类似 chat，但输入是单个 prompt 字符串

#### 4.2.3 `/v1/models`

返回可用模型列表

#### 4.2.4 `/v1/embeddings` (可选)

如果模型支持 embedding，返回向量

### 4.3 SSE Streaming

**任务**: 实现流式输出

**实现**:
```python
from fastapi.responses import StreamingResponse

async def stream_generator(engine, prompt):
    for token in engine.generate_stream(prompt):
        chunk = {
            "id": "chatcmpl-xxx",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": token}}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(stream_generator(...), media_type="text/event-stream")
```

### 4.4 认证与限流

**任务**: API key 认证 + 速率限制

**实现**: `qwen_infer/api/auth.py`

**认证**:
```python
from fastapi import Header, HTTPException

async def verify_api_key(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API key")
    api_key = authorization[7:]
    if api_key not in VALID_API_KEYS:
        raise HTTPException(401, "Invalid API key")
```

**限流**:
- 使用 `slowapi` 库
- 限制：100 req/min per API key

### 4.5 Usage 计量

**任务**: 统计 token 使用量

**实现**: `qwen_infer/api/usage_tracker.py`

**记录**:
- 每次请求的 prompt_tokens, completion_tokens
- 保存到 SQLite：`(timestamp, user_id, prompt_tokens, completion_tokens)`
- 提供查询接口：`GET /v1/usage?start_date=...&end_date=...`

### 4.6 Gateway 对接测试

**任务**: 验证与主流 Gateway 的兼容性

**测试对象**:
1. **OpenRouter** (https://openrouter.ai)
   - 注册自定义模型
   - 验证请求转发

2. **LiteLLM** (https://github.com/BerriAI/litellm)
   - 配置 custom endpoint
   - 测试 fallback 和 load balancing

3. **FastGPT** (https://github.com/labring/FastGPT)
   - 添加自定义模型
   - 验证对话流程

**验证脚本**: `tests/test_gateway_compat.py`

---

## Phase 5: Optimization & Paper

**目标**: 性能优化 + 论文撰写

### 5.1 Custom CUDA Kernels

**任务**: 用 CUDA 重写性能瓶颈

#### 5.1.1 Paged Attention Kernel

**优化点**:
- 当前 PyTorch 实现：逐 block gather → concat → attention
- CUDA 实现：直接在 kernel 内处理非连续 blocks，减少内存拷贝

**参考**: vLLM 的 `paged_attention_v2` kernel

**实现**: `qwen_infer/kernels/paged_attention.cu`

#### 5.1.2 MoE Dispatch Kernel

**优化点**:
- 当前：CPU 路由 → GPU 加载 expert → 计算
- CUDA 实现：GPU 端完成路由 + expert 并行调度

**关键**:
- 利用 shared memory 缓存 expert 权重
- Warp-level 并行处理不同 expert

**实现**: `qwen_infer/kernels/moe_dispatch.cu`

### 5.2 Speculative Decoding

**任务**: 用小模型加速大模型生成

**原理**:
1. 小模型（draft model）快速生成 K 个 token
2. 大模型（target model）并行验证这 K 个 token
3. 接受正确的前缀，拒绝后续

**MoE 特化**:
- Draft model = 只用 top-1 expert 的 Qwen（相当于 3B 模型）
- Target model = 完整 Qwen（35B，激活 3B）

**实现**: `qwen_infer/engine/speculative_decoding.py`

### 5.3 Continuous Batching

**任务**: 动态 batch，提高吞吐

**原理**:
- 传统 static batch：等所有序列都结束才开始下一个 batch
- Continuous batch：序列结束立即替换为新请求

**实现**: `qwen_infer/engine/continuous_batcher.py`

**关键**:
- 维护 `active_sequences` 队列
- 每步检查是否有序列结束 → 从 `pending_requests` 补充

### 5.4 Dynamic Batch Size

**任务**: 根据序列长度动态调整 batch size

**策略**:
- 短序列（<1k tokens）：batch_size = 16
- 中序列（1k-10k）：batch_size = 8
- 长序列（>10k）：batch_size = 2

**目标**: 最大化 GPU 利用率，避免 OOM

### 5.5 Prefix Caching

**任务**: 缓存公共前缀（如 system prompt）

**实现**:
- 检测多个请求的公共前缀
- 只计算一次，共享 KV cache blocks
- 结合 Persistent KV Cache（Phase 3）

### 5.6 性能对比实验

**任务**: 与 baseline 系统对比

**对比对象**:
1. vLLM (v0.4.0+)
2. llama.cpp (latest)
3. TGI (Text Generation Inference)
4. SGLang

**测试场景**:
1. **Throughput** (batch=1, 4, 8, 16)
2. **Latency** (TTFT, TPOT)
3. **Long context** (32k, 64k, 128k, 200k tokens)
4. **GPU utilization** (功耗 + 计算利用率)
5. **KV cache 复用** (跨会话恢复时间)

**硬件**:
- 8× V100 16GB (当前)
- 8× A100 40GB (如果有条件)

**脚本**: `benchmarks/compare_baselines.py`

### 5.7 论文撰写

**标题**: "MoE-Aware Inference with Persistent KV Cache: Optimizing Sparse Expert Activation for Long-Context LLMs"

**结构**:

1. **Abstract**
   - 问题：vLLM/llama.cpp 对 MoE 模型 GPU 利用率低
   - 方法：Expert 感知调度 + 持久化 KV cache
   - 结果：吞吐提升 X%，GPU 利用率 >80%

2. **Introduction**
   - MoE 模型的稀疏激活特性
   - 现有系统的局限性
   - 本文贡献

3. **Background**
   - MoE 架构
   - Paged Attention
   - NVLink 拓扑

4. **System Design**
   - Expert 调度策略
   - NVLink 拓扑感知
   - 持久化 KV cache 架构
   - 三级锁机制

5. **Implementation**
   - 自研量化
   - CUDA kernels
   - API 设计

6. **Evaluation**
   - 实验设置
   - 性能对比（表格 + 图表）
   - Ablation study（各优化的独立贡献）

7. **Related Work**
   - vLLM, SGLang, TGI
   - MoE 优化相关工作

8. **Conclusion**
   - 总结贡献
   - 未来工作（Ampere/Hopper 支持）

**投稿目标**: MLSys / OSDI / ATC

---

## 附录：开发优先级建议

**立即开始**:
1. Phase 1.1 (Bug 修复) — 1 天
2. Phase 1.2 (MoE 实现) — 1 周
3. Phase 1.3 (推理接入) — 3 天
4. Phase 1.4 (NVLink 拓扑) — 2 天

**并行开发** (Phase 1 完成后):
- Phase 2 (量化) — 1 人，2 周
- Phase 3 (KV cache) — 1 人，2 周

**后续串行**:
- Phase 4 (API) — 1 周
- Phase 5 (优化 + 论文) — 1 个月

**总工期估算**: 2-3 个月（1-2 人团队）
