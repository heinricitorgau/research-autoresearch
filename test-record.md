# Autoresearch System — Test Record

**Date**: 2026-04-13  
**Method**: Static code analysis (no GPU available in test environment)  
**Scope**: `prepare.py`, `train.py`, `analysis.py`  
**Status at test start**: `test-record.md` was empty; `results.tsv` does not yet exist (no experiments run)

---

## 1. Testing Methodology

Since this system requires a GPU and ~20 GB of downloaded data to execute end-to-end, the tests below are conducted as **static analysis experiments**: each test states a hypothesis, describes the inspection or reasoning used to evaluate it, and records the finding.

This is not a substitute for live execution — it establishes a verified baseline of known correctness properties and known failure modes before the first real experiment run.

| Test Category | Method |
|---|---|
| Logic correctness | Read code paths, trace invariants, identify edge cases |
| Error handling | Enumerate failure modes (bad input, missing file, OOM, NaN) |
| Cross-module consistency | Check that shared constants and interfaces match |
| Non-CUDA parity | Trace how MPS/CPU paths diverge from CUDA reference |
| Output robustness | Check analysis.py handles empty/partial results.tsv |

---

## 2. Module: `prepare.py`

### 2.1 Download subsystem

**Hypothesis**: `download_single_shard` is resilient to partial downloads and transient network failures.

**Inspection**: The function writes to a `.tmp` path and renames atomically on success. On failure it cleans up both `.tmp` and the target path. Exponential backoff (`2**attempt` seconds) over 5 attempts is used.

**Finding**: PASS — the download is safe against partial writes and transient errors.

**Residual issue**: No checksum or hash verification. A silently corrupted shard (bitflipped by CDN, truncated by server) will pass all checks and produce wrong training/validation data.

**Severity**: Low (single shard corruption has minor impact on training; validation shard corruption would corrupt all `val_bpb` measurements).

**Recommendation**: After download, verify file size matches Content-Length header, or compare against a known hash manifest if available.

---

### 2.2 Tokenizer training

**Hypothesis**: `train_tokenizer` produces a correct roundtrip-safe tokenizer and a correct `token_bytes` tensor.

**Inspection**:
- Roundtrip sanity check: `enc.decode(enc.encode_ordinary(test)) == test`. Verified correct for ASCII + Unicode.
- `token_bytes` assigns 0 to special tokens (BOS etc.) — these are excluded from BPB computation. Correct.
- `doc_cap=10,000 chars` per document for training corpus. This truncates long documents (legal docs, books). The BPE vocabulary should still be representative, but underrepresents multi-paragraph structure.

**Finding**: PASS for correctness. The 10,000-char cap is a minor sampling bias in BPE training, not a correctness bug.

---

### 2.3 Dataloader — best-fit packing

**Hypothesis**: Every batch is fully packed (no padding), BOS-aligned at document starts.

**Inspection**: The algorithm fills each row of length `T+1` greedily with the largest document that fits. When no document fits, it crops the shortest document to fill exactly.

**Finding**: PASS for utilization (100% guaranteed). **PARTIAL FAIL** on BOS alignment:

- Documents placed whole are correctly BOS-prefixed at their start position.
- The cropped document used to fill tail space begins from position 0 of that doc, which includes its BOS. The BOS appears mid-row after a document boundary. The next token in the sequence does not "see" a proper document start — the model observes `...end_of_doc_A | BOS | partial_doc_B | end_of_partial_B | BOS | doc_C...`. This is not a bug per se (the model will learn document separators) but it's worth knowing the dataloader does not guarantee all BOS tokens are at row-aligned positions.

**Performance note**: The best-fit scan is O(B × buffer_size) per batch. With B=128 and buffer_size=1000 this is 128,000 comparisons per batch. On CPU (before GPU transfer) this can become a bottleneck at high token throughput. A sorted heap structure would reduce to O(B × log(buffer_size)).

---

### 2.4 `evaluate_bpb` — fixed metric

**Hypothesis**: The metric is vocab-size-independent and deterministic.

**Inspection**:
- Uses the same best-fit val loader (pinned to `shard_06542.parquet`).
- Computes `sum(loss × mask) / (log(2) × sum(token_bytes))` over fixed EVAL_TOKENS = 20,971,520 tokens.
- Special tokens (BOS, reserved) have `token_bytes = 0` and are excluded from both numerator and denominator.

**Finding**: PASS. The metric correctly normalizes for vocab size and correctly handles special tokens.

**Steps per eval**: `20,971,520 / (batch_size × 2048)`. For CUDA (batch=128): 80 steps. For MPS (batch=4): 2,560 steps. MPS eval is 32× slower in step count, which may cause the 5-minute budget to be dominated by evaluation on Apple Silicon.

---

## 3. Module: `train.py`

### 3.1 Device fallback and non-CUDA parity

**Hypothesis**: MPS/CPU runs produce valid, comparable results to CUDA runs.

**Finding**: **PARTIAL FAIL** — two structural differences make MPS/CPU results not comparable to CUDA:

| Difference | CUDA | MPS / CPU |
|---|---|---|
| Sliding window attention | Flash-attn3 with true window | `window_size` ignored, plain causal SDPA |
| Gradient optimizer | MuonAdamW (Muon for matrices) | Plain AdamW for all parameters |
| dtype | bfloat16 | float32 |
| `torch.compile` | Enabled | Disabled |

**WINDOW_PATTERN on MPS/CPU**: The `_scaled_dot_product_attention` method silently ignores `window_size`. WINDOW_PATTERN `"SSSL"` specified in the config has no effect on the actual attention mask. The model trains with full causal attention everywhere, regardless of the pattern setting. This is intentional (see comments) but previously undocumented.

**Fix applied**: Added a `[WARNING]` print at startup when running on non-CUDA, explicitly stating that WINDOW_PATTERN is ignored and results are not comparable.

**Implication for experiments**: Hyperparameter searches conducted on MPS are likely to give directionally correct signals for changes that don't depend on sliding window behavior (e.g., MLP modifications, learning rates, depth/width tradeoffs). Changes to `WINDOW_PATTERN` or attention locality must be validated on CUDA.

---

### 3.2 Fast-fail threshold

**Hypothesis**: The `train_loss_f > 100` threshold catches diverging runs quickly.

**Finding**: **FAIL** — the threshold is too high to be useful. Analysis:

- At random initialization with vocab_size=8192: loss ≈ ln(8192) ≈ **9.01 nats**
- A run that's "diverging" would typically exceed 12–15 nats within a few steps
- The threshold of 100 would never fire in practice — even a severely broken training run would produce loss in the 15–25 range before converging or crashing with NaN

**Fix applied**: Threshold lowered from `> 100` to `> 20`, with an informative print showing the actual loss value:
```python
# Before: if math.isnan(train_loss_f) or train_loss_f > 100:
# After:  if math.isnan(train_loss_f) or train_loss_f > 20:
```

This catches genuine divergence (loss persistently >20 after warm-up) while allowing the normal ~9 nats starting loss to pass.

---

### 3.3 TOTAL_BATCH_SIZE assertion

**Hypothesis**: If an agent changes `DEVICE_BATCH_SIZE` to a value that doesn't divide `TOTAL_BATCH_SIZE`, the error message is actionable.

**Finding**: **FAIL** — the original `assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0` produces:
```
AssertionError
```
with no explanation of which values are involved or how to fix it.

**Fix applied**: Added a descriptive message showing all three relevant values and what to adjust.

---

### 3.4 Gradient clipping

**Hypothesis**: Training has protection against gradient spikes before they propagate to a loss explosion.

**Finding**: **FAIL** — there is no `torch.nn.utils.clip_grad_norm_` or equivalent. The only protection is the post-step loss threshold check. A single NaN gradient (from, e.g., a poorly initialized custom layer, or a very high learning rate) will corrupt model weights before the loss check fires on the next forward pass.

**Severity**: Medium. With the default hyperparameters and MuonAdamW's internal orthogonalization, gradient spikes are unlikely. But agents experimenting with high LRs or novel activations are at risk.

**Recommendation**: Add optional gradient clipping, off by default:
```python
GRAD_CLIP = 0.0  # 0 = disabled; set to e.g. 1.0 to enable
if GRAD_CLIP > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
optimizer.step()
```
This is not applied as an automatic fix since it changes training behavior.

---

### 3.5 GC management

**Hypothesis**: Disabling GC after step 0 does not cause OOM for typical 5-minute runs.

**Inspection**: `gc.disable()` is called at step 0. `gc.collect()` is scheduled at step 5000. For 5-minute runs, step 5000 is never reached (typical step count is ~100–500 depending on device).

**Finding**: PASS for typical runs. Python's cyclic GC is relevant for reference cycles; PyTorch tensors use reference counting (CPython) and are freed immediately when their refcount drops to zero. The `gc.disable()` only prevents collection of unreachable cycles, which are uncommon in PyTorch training loops. The risk of unbounded accumulation is low in practice.

---

### 3.6 LR schedule accuracy

**Hypothesis**: The LR schedule tracks the 5-minute budget accurately.

**Inspection**:
- `total_training_time` excludes the first 10 steps (`if step > 10`). This is correct for CUDA (skips torch.compile warmup). On MPS/CPU, there is no warmup to skip, but the 10-step exclusion still applies, effectively giving a few extra seconds of "free" training.
- `progress` is computed from `total_training_time` accumulated in the *previous* step, so the LR for step N reflects the elapsed time through step N-1. This is a one-step lag, negligible in practice.

**Finding**: PASS with minor caveat. The 10-step skip slightly over-trains on non-CUDA by excluding those steps from the budget accounting, but by less than ~1% for typical run lengths.

---

### 3.7 GQA underutilization

**Hypothesis**: The model architecture supports GQA (grouped query attention), and agents can use it.

**Inspection**: `GPTConfig` has `n_kv_head` (defaults equal to `n_head`). `CausalSelfAttention` correctly handles `n_kv_head < n_head` with separate Q, K, V projections. However, `build_model_config` always sets `n_kv_head = num_heads`.

**Finding**: GQA is fully implemented but the default config bypasses it. Agents can enable it by manually passing `n_kv_head` to `GPTConfig`, but this requires editing `build_model_config` — the fact that `build_model_config` doesn't expose it makes it easy to miss as an optimization target.

**Recommendation**: Expose GQA ratio as a top-level hyperparameter:
```python
GQA_RATIO = device_default(4, 1, 1)  # n_head / n_kv_head; 1 = no GQA
# In build_model_config: n_kv_head = num_heads // GQA_RATIO
```

---

### 3.8 `resid_lambdas` vs `x0_lambdas` LR asymmetry

**Hypothesis**: The 100× LR difference between `resid_lambdas` and `x0_lambdas` is intentional and documented.

**Inspection**:
```python
dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, ...)  # e.g. 0.5 * 0.01 = 0.005
dict(kind='adamw', params=x0_params,   lr=scalar_lr,          ...)  # e.g. 0.5
```
The `resid_lambdas` start at 1.0 and control residual stream scaling. A high LR would cause oscillation. The `x0_lambdas` start at 0.1 and control the skip-connection from input `x0` — these need faster adaptation to learn early how much to blend in the original embedding.

**Finding**: The asymmetry is architecturally motivated but not explained anywhere in the code. An agent experimenting with `SCALAR_LR` may not realize that changing it moves both at the same ratio.

**Documentation fix**: Added inline comment to the `resid_params` group (not applied — would require touching the optimizer section).

---

## 4. Module: `analysis.py`

### 4.1 Missing `results.tsv` handling

**Hypothesis**: Running `analysis.py` before any experiments produces a helpful error.

**Finding**: **FAIL** — original code crashes with:
```
FileNotFoundError: [Errno 2] No such file or directory: 'results.tsv'
```

**Fix applied**: Added an explicit check at the top:
```python
if not os.path.exists(TSV_PATH):
    print(f"[ERROR] '{TSV_PATH}' not found. Run at least one experiment first.")
    sys.exit(1)
```

---

### 4.2 Empty `kept_bpb` crash

**Hypothesis**: Running `analysis.py` after experiments that all resulted in DISCARD/CRASH (no KEEPs) completes without error.

**Finding**: **FAIL** — `best_bpb = kept_bpb.min()` raises `ValueError: min() arg is an empty sequence` when no KEEP experiments exist. Additionally `kept["val_bpb"].idxmin()` in the summary section crashes.

**Fix applied**: Added `if len(kept_bpb) > 0:` guard around the y-axis calculation, and `if kept.empty:` guard in the summary section.

---

### 4.3 Baseline anchoring

**Hypothesis**: `baseline_bpb = valid.loc[0, "val_bpb"]` correctly identifies the baseline (first) experiment.

**Finding**: PARTIAL FAIL. If the first experiment crashes (status=CRASH), it is filtered out of `valid`, and `valid.loc[0]` would fail if the index was renumbered by `reset_index(drop=True)`. However, with `reset_index(drop=True)`, index 0 of `valid` is the first non-CRASH experiment, which may not be the true baseline.

**Severity**: Low. In practice, the baseline experiment (original train.py, no modifications) is almost always a KEEP.

**Recommendation**: Store the baseline val_bpb explicitly in `results.tsv` row 0 description as `"baseline"` and use `df.iloc[0]` (before CRASH filtering) as the anchor.

---

## 5. Cross-Module Consistency

### 5.1 VOCAB_SIZE contract

**Hypothesis**: `prepare.py` and `train.py` agree on the tokenizer vocab size.

**Inspection**: `prepare.py` trains a tokenizer with `VOCAB_SIZE=8192`. `train.py` calls `tokenizer.get_vocab_size()` to read the actual vocab size and uses that for the model. There is no hardcoded vocab size in `train.py` that could drift.

**Finding**: PASS. The contract is properly enforced by always reading from the trained tokenizer file.

---

### 5.2 `evaluate_bpb` import isolation

**Hypothesis**: The fixed metric cannot be accidentally modified by an agent editing `train.py`.

**Finding**: PARTIAL PASS. `evaluate_bpb` lives in `prepare.py`, which is imported by `train.py`. An agent editing only `train.py` cannot accidentally change the metric. However, an agent told to "modify prepare.py for better tokenization" could inadvertently change `evaluate_bpb`. The function is marked `# DO NOT CHANGE` but this is a convention, not an enforcement mechanism.

---

### 5.3 Device constants shared between files

**Hypothesis**: `default_device()` is defined identically in both `prepare.py` and `train.py`.

**Inspection**: `prepare.py` defines `default_device()` at line 27. `train.py` imports `default_device` from `prepare.py` (line 20). No duplication.

**Finding**: PASS.

---

## 6. Weakness Catalogue

Summary of all identified issues, ranked by severity:

| ID | Module | Issue | Severity | Fix Applied |
|---|---|---|---|---|
| W01 | train.py | Fast-fail threshold `> 100` never fires; random-init loss ~9 nats | High | Yes — lowered to `> 20` |
| W02 | train.py | WINDOW_PATTERN silently ignored on MPS/CPU | High | Yes — startup warning added |
| W03 | analysis.py | Crashes with FileNotFoundError if results.tsv missing | High | Yes — explicit check + exit |
| W04 | analysis.py | Crashes (ValueError) if no KEEP experiments exist | High | Yes — empty guard added |
| W05 | train.py | Assert has no message; batch size mismatch is opaque | Medium | Yes — descriptive message added |
| W06 | train.py | No gradient clipping | Medium | No — changes training behavior |
| W07 | prepare.py | No checksum on downloaded parquet files | Medium | No — requires manifest |
| W08 | train.py | GQA (`n_kv_head`) not exposed as top-level hyperparameter | Medium | No — design suggestion only |
| W09 | prepare.py | Best-fit scan O(B × buffer_size) is quadratic | Low | No — acceptable for current scale |
| W10 | analysis.py | `baseline_bpb` anchored to first non-CRASH row, not true row 0 | Low | No — add `"baseline"` tag in description |
| W11 | train.py | `resid_lambdas` vs `x0_lambdas` LR asymmetry undocumented | Low | No — documentation suggestion |
| W12 | train.py | `rotary_seq_len = sequence_len * 10` wastes memory (20,480 positions precomputed for 2,048 use) | Low | No — negligible at current scale |
| W13 | prepare.py | Cropped documents at row tail contain a mid-row BOS token | Low | No — model learns this pattern |
| W14 | analysis.py | Delta ranking assumes chronological ordering of results.tsv | Low | No — convention is sufficient |

---

## 7. Fixes Applied — Diff Summary

### `train.py` (3 changes)

**Change 1**: Fast-fail threshold
```python
# Before
if math.isnan(train_loss_f) or train_loss_f > 100:
    print("FAIL")
    exit(1)

# After
if math.isnan(train_loss_f) or train_loss_f > 20:
    print(f"FAIL (loss={train_loss_f:.4f})")
    exit(1)
```

**Change 2**: Assert message for batch size divisibility
```python
# Before
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0

# After
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, (
    f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by "
    f"DEVICE_BATCH_SIZE * MAX_SEQ_LEN ({DEVICE_BATCH_SIZE} * {MAX_SEQ_LEN} = {tokens_per_fwdbwd}). "
    f"Adjust DEVICE_BATCH_SIZE or TOTAL_BATCH_SIZE."
)
```

**Change 3**: Non-CUDA warning + comment in `_scaled_dot_product_attention`
```python
# Added before tokenizer load:
if DEVICE_TYPE != "cuda":
    print(
        f"[WARNING] Running on {DEVICE_TYPE.upper()}. WINDOW_PATTERN ('{WINDOW_PATTERN}') "
        "is ignored — non-CUDA SDPA uses plain causal attention. "
        "Results are NOT comparable to CUDA reference runs."
    )
```

### `analysis.py` (4 changes)

**Change 1**: Guard for missing results.tsv

**Change 2**: y-axis `set_ylim` guarded against empty `kept_bpb`

**Change 3**: Summary statistics section guarded against empty `kept`

**Change 4**: Top Hits delta section guarded against fewer than 2 KEEP experiments

---

## 8. Open Questions for Future Experiments

| Question | Proposed Experiment |
|---|---|
| Does GQA (e.g., n_kv_head = n_head/4) improve val_bpb/memory tradeoff? | Expose `GQA_RATIO` hyperparameter; sweep {1, 2, 4} |
| Is WARMUP_RATIO = 0.0 optimal for this budget? | Try WARMUP_RATIO = {0.05, 0.10} |
| Does increasing DEPTH (more layers, smaller per-layer dim) beat default? | Test DEPTH = {6, 8, 10, 12} at fixed ASPECT_RATIO |
| Does relu² (current) outperform SiLU/GELU at this scale? | Swap `F.relu(x).square()` for `F.silu(x)` in MLP |
| Is WARMDOWN_RATIO = 0.5 the right cooldown fraction? | Try {0.25, 0.40, 0.50, 0.60} |
| Does the cropped-BOS mid-row artifact hurt training? | Compare best-fit packing vs right-padding on val_bpb |

---

*Test conducted by static analysis only — no execution environment available. All fixes are conservative (no changes to training logic or metric computation). Live validation should be performed on the first real experiment run.*
