# V100 (sm_70) 内核调校 for llama.cpp 😊 (FOR ENGLISH VER PLEASE SCROLL DOWN😊)

> **长上下文解码：90K 快 24%，262K 快 52%；预填充快 36~57%。**
> 同一张卡、同一个模型，上游 vs 本分支。

针对 **Tesla V100-SXM2 32GB** 的推理内核调优，专治长上下文这个最难受的场景。

上游 llama.cpp 在 Volta 上留了不少余量：解码时注意力每生成一个 token 都要重新转换一遍 KV
缓存，预填充的注意力内核又按 Ampere 的分块形状调的，不适合 sm_70。这个分支把这些补上，
另外做了几处小算子融合。

## 实测结果

测试方式：`llama-bench`，单张 V100 32G，模型 **Qwen3.8-27B Q4_K_M**（48 层 SSM + 16 层
全注意力），`-fa on`，q8_0 KV；解码用 tg128，预填充用 pp512。MTP 那行是真实
8.8 万 token 提示词的端到端结果：

😊

预填充再加上 `-b 4096 -ub 4096`，90K 深度从 501 提到 **726 t/s**（见下）。

## 改了什么

* **解码注意力直读 q8_0 KV 缓存。** 在 shared memory 的分块内直接反量化，省掉每个 token
  先把整份缓存转成 f16 的开销；共用一个 KV 头的 6 个 query 头共用一块 tile。这是长上下文
  收益的大头。
* **预填充注意力换成 Volta 专属分块配置。** 上游把 Ampere 的配置直接用在 Volta 上，会溢出
  寄存器。把 Q 挪出寄存器、缩小载入缓冲之后，每个 SM 能跑 2 个 block 而不是 1 个。
* **小算子融合。** 同一次前向里只算一次激活量化（原本每个矩阵乘各算一遍）、
  `[rms_norm, scale]` 合成一个内核、权重侧换成 Volta 专属参数表。

另外两个「零代码」发现：

* `-b 4096 -ub 4096` 能把 90K 深度的预填充从 501 提到 **726 t/s**（分块预填充每块都要
  重读一遍 KV 缓存）。
* **f16 KV 比 q8_0 快约 5%**（约 24 万上下文以内）。开到 262144 时 f16 缓存要占约 16 GB，
  所以默认还是 q8_0 更稳妥。

## 编译

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

## 运行时开关

每一处改动都能关掉，方便你自己对照测试：

| 环境变量 | 作用 |
|---|---|
| `GGML_V100_UPSTREAM_BUILD=1` | 恢复上游的注意力实现 |
| `GGML_V100_FA=vec\|tile\|mma` | 强制指定 flash-attention 内核 |
| `GGML_V100_MMVQ_Q8_1=0` | 关闭激活量化缓存 |
| `GGML_V100_RMS_SCALE=0` | 关闭 rms_norm + scale 融合 |
| `GGML_V100_SSM=1` | 打开 SSM epilogue 融合（默认关，只 +0.3%） |

## 注意

* 这些提升是 V100 + 这个混合架构模型特有的，换成稠密模型或别的架构，数字会小很多——
  重点本来就是长上下文。
* 分块相关的改动会改变浮点累加顺序，所以输出可能与上游存在最低位的差异。长上下文大海捞针
  测试和 90K 验收跑下来没有质量退化。
* 分支基于上游提交 `df03399b`（其 CUDA 源码与本次开发和实测所用的源码逐字节一致）。
  改动量：**11 个文件，+930 / −86**。

# V100 (sm_70) kernel tuning for llama.cpp 😊

> **Long-context decode: +24% at 90K, +52% at 262K. Prefill: +36~57%.**
> Same card, same model, upstream vs this branch.

Kernel-level tuning for the **Tesla V100-SXM2 32GB**, for the case that actually hurts:
long-context decoding.

Upstream llama.cpp leaves a lot on the table on Volta. Decode attention re-formats the
KV cache on every token, and the prefill attention kernels are tuned for Ampere tile
shapes, not for sm_70. This branch fixes those, and fuses a few small operators.

## Results

Measured with `llama-bench` on one V100 32G running **Qwen3.8-27B Q4_K_M**
(48 SSM + 16 full-attention layers), `-fa on`, q8_0 KV; decode is `tg128`, prefill is
`pp512`. The MTP row is a real end-to-end 88K-token prompt:

😊


Prefill with `-b 4096 -ub 4096` reaches **726 t/s** at 90K depth, against 501 with the
default batch size (see below).

## What changed

* **Decode attention reads the q8_0 KV cache directly.** The tile kernel dequantises
  inside shared memory instead of converting the whole cache first, and six query heads
  that share a KV head share one tile. This is the big long-context win.
* **Prefill attention gets Volta-specific tile configs.** Upstream ran Ampere configs on
  Volta, which spilled registers. Moving Q out of registers and shrinking the load
  buffers lets two blocks fit per SM instead of one.
* **Small operators are fused.** The activation quantisation is computed once per graph
  pass instead of once per matvec, `[rms_norm, scale]` becomes a single kernel, and the
  MMVQ weight path gets a Volta-specific table.

Two findings that need no code at all:

* `-b 4096 -ub 4096` takes prefill from 501 to **726 t/s** at 90K depth (chunked prefill
  re-reads the KV cache once per chunk).
* **f16 KV is ~5% faster than q8_0** for decoding up to ~240K context. At 262144 the f16
  cache needs ~16 GB, so q8_0 stays the safe default.

## Build

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

## Runtime switches

Every change can be turned off, so you can A/B it yourself:

| env var | effect |
|---|---|
| `GGML_V100_UPSTREAM_BUILD=1` | restore upstream attention behaviour |
| `GGML_V100_FA=vec\|tile\|mma` | force a flash-attention kernel |
| `GGML_V100_MMVQ_Q8_1=0` | disable the activation-quantisation cache |
| `GGML_V100_RMS_SCALE=0` | disable the rms_norm + scale fusion |
| `GGML_V100_SSM=1` | enable the SSM epilogue fusion (off by default, +0.3%) |

## Notes

* These numbers are specific to the V100 plus this hybrid model. Dense models and other
  architectures will see smaller gains — the long-context effect is the point.
* The tiling changes alter the floating-point accumulation order, so output can differ
  from upstream in the last bits. Long-context needle tests and a 90K acceptance run
  showed no quality regression.
* Based on upstream commit `df03399b` (CUDA sources byte-identical to the tree this was
  developed and measured against). Diff: **11 files, +930 / −86**.

---


