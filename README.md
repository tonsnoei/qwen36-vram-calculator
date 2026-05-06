# VRAM Calculator - Qwen3.6 & Gemma-2-27B

Inference VRAM estimation for three specific models, with correct handling of hybrid attention and Mixture-of-Experts.

![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![No backend](https://img.shields.io/badge/backend-none-lightgrey?style=flat-square)

---

## Try it now

[![Try it now](https://img.shields.io/badge/Try%20it%20now-Live%20Demo-6366f1?style=for-the-badge&logo=github)](https://tonsnoei.github.io/qwen36-vram-calculator/)

---

## Why a specific calculator

Generic VRAM formulas assume a model consists of classical transformer layers with grouped-query attention. That holds for Gemma-2 and the Qwen2.5 series, but **not** for Qwen3.6.

Qwen3.6 uses a **hybrid architecture**: of the 64 layers in Qwen3.6-27B, only 16 are classical attention layers; the other 48 are linear-attention layers (Gated DeltaNet) that have no sequence-dependent KV cache. A naive calculator therefore overestimates the KV cache of Qwen3.6-27B by roughly a factor of 5.8.

In addition, Qwen3.6-35B-A3B is a **MoE model**: 35B total parameters but only 3B active per token. All weights must reside in VRAM (otherwise expert routing is slow), but activations scale with the active parameters, not the total.

This calculator handles each of those cases separately.

---

## Which models

Three models, no size slider - the specs are hardcoded from the official model cards.

| Model | Type | Total | Active | Layers | Attention layers | KV heads × dim |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | dense, hybrid | 27B | 27B | 64 | 16 | 4 × 256 |
| Qwen3.6-35B-A3B | MoE, hybrid | 35B | 3B | 40 | 10 | 2 × 256 |
| Gemma-2-27B | dense, classical | 27B | 27B | 46 | 46 | 16 × 128 |

Sources: [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Gemma-2 paper (arxiv 2408.00118)](https://arxiv.org/pdf/2408.00118).

---

## The six components

The total is always the sum of six components, each described below.

### 1. Bytes per parameter

```
Q4   → 0.55 bytes/param   (4-bit + ~10% metadata overhead)
Q8   → 1.0  bytes/param
FP16 → 2.0  bytes/param
FP32 → 4.0  bytes/param
```

The 0.55 for Q4 covers the typical metadata of AWQ, GPTQ-INT4, and GGUF Q4_K_M (scales and zero-points per block of 32 elements). Pure 4-bit would be 0.5; the 10% extra is empirical and consistent across common formats.

### 2. Model weights

```
weights_GB = totalParams × bytesPerParam / 1024³
```

Important for MoE: use the **total** parameter count, not the active count. All experts must reside in VRAM; otherwise the system has to fetch them from CPU/disk during routing, breaking latency. For Qwen3.6-35B-A3B Q8 that means 32.6 GB, not 2.8 GB.

### 3. KV cache (classical attention layers)

```
KV_worst_case = 2 × L_attn × N_kv × D_head × S × B × C_b / 1024³
KV_realistic  = KV_worst_case × 0.60
```

Where:
- `L_attn` - number of **classical** attention layers (not the total layer count!)
- `N_kv` - KV heads
- `D_head` - head dimension
- `S` - sequence length
- `B` - batch size = `max(batch_input, users_input)`
- `C_b` - bytes per KV element (FP16 = 2, INT8/FP8 = 1)
- factor 2 - K cache + V cache

The 0.60 factor models PagedAttention/vLLM allocating KV blocks on demand rather than upfront. With continuous batching and mixed sequence lengths, roughly 60% of slots are filled; worst case only occurs when all sequences simultaneously reach the full length `S`.

**The crucial difference for hybrid models:** for Qwen3.6-27B, `L_attn = 16`, not 64. For Gemma-2-27B, `L_attn = 46` (all layers classical). This is why the same sequence length produces a much smaller KV cache on Qwen3.6.

### 4. Linear-attention state (hybrid models only)

```
linear_state_GB = L_linear × state_per_layer × B / 1024³
```

Gated DeltaNet maintains a recurrent state per layer per sequence of size `V_heads × head_dim × head_dim`. For Qwen3.6-27B: 48 V-heads × 128 × 128 × 2 bytes ≈ 1.5 MB per layer per sequence.

Key property: this state is **constant in sequence length**. This is why Qwen3.6 can handle 256k context without memory spiraling out of control - only the 16 classical layers scale with `S`; the other 48 do not.

In practice this component stays below 1 GB and never dominates. It is shown as a separate line in the breakdown to make the difference from classical models explicit.

### 5. Activations

```
seq_factor    = max(1, log₂(S / 4096) + 1)
active_factor = max(0.3, activeParams / 27)
activations_GB = min(1 + B × 0.1 × seq_factor × active_factor, 12)
```

This is an empirical estimate that scales with:
- batch size (linear)
- sequence length (logarithmic - not linear, because activations are computed layer by layer, not all held in memory at once as in training)
- active parameters (linear relative to a 27B baseline)

Capped at 12 GB to prevent extreme combinations (70B model, 64k seq, batch 32) from producing unrealistic numbers. For MoE the **active** parameter count is used: Qwen3.6-35B-A3B with 3B active gets `active_factor = 0.3`, making it significantly lighter than the same inputs on a 27B dense model.

### 6. Framework overhead

```
overhead_GB = 1.5 + ⌊totalParams / 30⌋ × 0.5
            + 1.0  if MoE
```

CUDA workspace, cuBLAS handles, attention kernel scratchpads, and for MoE the routing tables and gate cache. The 1.5 GB base is what a minimal vLLM/SGLang server claims before a single request arrives; the scaling factor captures the fact that larger models need more workspace for matmul tiling.

### Total

```
total_GB = weights + KV_realistic + linear_state + activations + overhead
```

The calculator also shows the KV cache worst case separately, so you can see what happens when continuous batching is disabled or when all users simultaneously run long sequences.

---

## Example calculation: Qwen3.6-27B, Q8, batch 12, 16k seq, FP16 KV

```
weights        = 27 × 10⁹ × 1.0 / 1024³                      = 25.15 GB
KV worst case  = 2 × 16 × 4 × 256 × 16000 × 12 × 2 / 1024³   = 11.72 GB
KV realistic   = 11.72 × 0.60                                  =  7.03 GB
linear state   = 48 × (48×128×128×2) × 12 / 1024³             =  0.84 GB
activations    = min(1 + 12 × 0.1 × 2.97 × 1, 12)             =  4.56 GB
overhead       = 1.5 + ⌊27/30⌋ × 0.5                          =  1.50 GB
────────────────────────────────────────────────────────────────────────
total                                                          = 39.08 GB
```

Fits on an RTX 6000 Ada (48 GB) with tight headroom. The same inputs on Gemma-2-27B → 71.6 GB (requires an 80 GB GPU) purely due to a KV cache that is 5.8× larger.

---

## GPU fit assessment

The table on the right shows VRAM utilization percentage and a status label for each GPU:

| Utilization | Status | Meaning |
|---|---|---|
| ≤ 70% | Comfortable | Plenty of room for peak-load fluctuations |
| 70-90% | Tight | Works, but no margin for unexpected spikes |
| 90-100% | Marginal | Very risky - OOM with an unfavorable request mix |
| > 100% | Insufficient | Does not fit |

The verdict line at the bottom uses the same thresholds (70% comfortable, 95% of a 96 GB GPU as upper bound) so that the table and verdict remain consistent.

---

## Important caveats

**Hybrid layer distribution is idealized.** Qwen3.6 alternates classical and linear layers in a 1:3 pattern (1 attention per 3 DeltaNet). This is what the model card states, but the exact distribution may change between releases - check `config.json` if precise figures are needed.

**DeltaNet state size is an estimate.** The ~1.5 MB per layer per sequence follows from the reported V-head/head-dim configuration and the standard recurrent-state structure of linear attention. The actual allocation in vLLM or SGLang may differ slightly due to padding and alignment, but within 1 GB the difference is negligible.

**MoE weight quantization is uniform.** The calculator applies `bytes_per_param` to all 35B parameters. In practice some MoE runtimes (KTransformers) use different quantization for experts versus shared layers; this is not modeled here.

**Tensor parallelism not included.** With multi-GPU setups (TP-2, TP-8) the actual per-GPU footprint is smaller than the estimate, but communication overhead is added. The calculator estimates the single-GPU equivalent only.

**Activations are empirical.** The formula is fitted to measured values for batch 1-32, sequence 4k-64k, 7B-70B models. Outside that range (very short sequences, very small batches) the estimate is less accurate - overhead then sets the lower bound.

**GiB versus GB.** All calculations divide by 1024³, so technically the unit is GiB. The label reads "GB", consistent with how HuggingFace and vLLM report their figures. For a 70B Q8 model the difference between 70 GB (decimal) and 65.2 GiB is roughly 5 GB.

---

## Sources

- Qwen3.6-27B specs: <https://huggingface.co/Qwen/Qwen3.6-27B> (April 2026 release)
- Qwen3.6-35B-A3B specs: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- Gemma-2-27B specs: <https://arxiv.org/pdf/2408.00118> (Table 1)
- KV cache formula + PagedAttention fill factor: vLLM documentation
- Q4 metadata overhead: GGUF spec, AWQ paper, GPTQ paper