# Precomputed ESM-C `lm_z` Plan

## Motivation

Stage 1 MD fine-tuning currently freezes almost all of ESMFold2, but every train
step can still run frozen ESM-C before the pair trunk. In the local model this
happens in `ESMFold2Model._forward_impl(...)`:

```text
input_ids -> _compute_lm_hidden_states(...) -> language_model(...) -> lm_z
```

Then MD conditioning is added into the same pair tensor:

```text
lm_z = lm_z + md_z
```

Because Stage 1 only trains `md_conditioning`, the final ESM-C-derived `lm_z`
is constant for a given sequence, ESMFold2 checkpoint, ESM-C checkpoint,
precision mode, and code path. Precomputing it should remove the repeated ESM-C
forward pass from MD training.

## Current Tensor Shapes

The relevant local config is:

```text
d_pair = 256
lm_d_model = 2560
lm_num_layers = 80
```

`LanguageModelShim.forward(...)` takes:

```text
lm_hidden_states: [B, L, 81, 2560]
```

and returns:

```text
lm_z: [B, L, L, 256]
```

The cache target should be final `lm_z`, not raw hidden states, for this Stage 1
setup. If the ESMFold2 `language_model` projection is ever unfrozen, final
`lm_z` is no longer a valid static cache target; at that point cache hidden
states instead or keep computing live.

## Disk Estimate

I scanned the local mdCATH subset at:

```text
data/mdcath_320K_len_le200/data
```

Metadata read:

```text
domains = 4482
unique_sequences = 4482
min_L = 50
median_L = 102
mean_L = 108.32
q95_L = 182
max_L = 200
sum_L = 485487
sum_L2 = 59451793
```

For final `lm_z`, storage is:

```text
sum_L2 * 256 * bytes_per_element
```

Concrete estimates:

```text
lm_z bf16/fp16: 28.35 GiB / 30.44 GB
lm_z fp32:      56.70 GiB / 60.88 GB
lm_z int8:      14.17 GiB / 15.22 GB
```

For comparison, caching ESM-C hidden states is:

```text
sum_L * 81 * 2560 * bytes_per_element
```

Concrete estimates:

```text
hidden states bf16/fp16: 187.51 GiB / 201.34 GB
hidden states fp32:      375.03 GiB / 402.68 GB
```

So final `lm_z` in bf16 is about 6.6x smaller than bf16 hidden states on this
dataset. The existing mdCATH HDF5 directory is about 210G, ESMFold2 is about
1.3G, and local ESM-C-6B is about 24G.

## Cache Format

Use one unpadded cache file per domain or sequence:

```text
lm_z_cache/
  manifest.json
  index.jsonl
  tensors/
    153lA00.<sequence_sha256>.safetensors
```

Each tensor file stores:

```text
lm_z: [L, L, 256]
```

Use `safetensors` rather than `torch.save` so the cache is non-pickle, dtype
explicit, and simple to hash. Store the tensor unpadded; the training batch
collator should pad it to `[B, max_L, max_L, 256]`.

The manifest should include:

```text
cache_kind = "esmfold2_lm_z"
dtype = "bfloat16" or "float32"
model_path / model_id
esmc_path / esmc_id
esmfold2_config_hash
esmfold2_language_model_state_hash
esmc_state_hash or snapshot revision
transformers_source_git_sha
lm_mask_pct = 0.0
esmc_precision = "bf16" or "fp32"
torch_version
cuda_version
device_name
deterministic_flags
```

The index should include at least:

```text
domain
sequence
sequence_sha256
length
relative_tensor_path
tensor_sha256
```

## Model Changes

Add `lm_z` as a first-class optional input beside `lm_hidden_states`:

```python
def _forward_impl(..., input_ids=None, lm_hidden_states=None, lm_z=None, ...):
```

Then change the current LM block conceptually from:

```python
if lm_hidden_states is None and input_ids is not None and self._esmc is not None:
    lm_hidden_states = self._compute_lm_hidden_states(...)
lm_z = None
if lm_hidden_states is not None:
    lm_z = self.language_model(lm_hidden_states.detach())
```

to:

```python
if lm_z is None:
    if lm_hidden_states is None and input_ids is not None and self._esmc is not None:
        lm_hidden_states = self._compute_lm_hidden_states(...)
    if lm_hidden_states is not None:
        lm_z = self.language_model(lm_hidden_states.detach())
else:
    validate_lm_z_shape(lm_z, token_attention_mask)
```

Thread `lm_z` through:

```text
ESMFold2Model.forward(...)
ESMFold2Model.forward_train(...)
ESMFold2Model._forward_impl(...)
```

Keep the MD-conditioning merge exactly where it is now:

```python
lm_z = md_z if lm_z is None else lm_z + md_z.to(dtype=lm_z.dtype)
```

Also add a helper used by both precompute and tests:

```python
@torch.inference_mode()
def compute_lm_z(..., lm_mask_pct: float = 0.0) -> Tensor:
    hidden = self._compute_lm_hidden_states(...)
    return self.language_model(hidden.detach())
```

That avoids duplicating subtle ESM-C wrapping/collapsing behavior in the cache
script.

## Training Script Changes

Add arguments:

```text
--lm-z-cache-dir PATH
--require-lm-z-cache
--lm-z-cache-dtype bf16|fp32
--precomputed-lm-z-only
```

Minimal behavior:

- If `--lm-z-cache-dir` is set, `tensorize_record(...)` loads the cached tensor
  for the record sequence and inserts `features["lm_z"]`.
- `pad_features(...)` handles `lm_z` as a pair key with shape
  `[1, max_L, max_L, 256]`.
- `training_loss(...)` remains unchanged because it already forwards `**features`.
- `load_model(...)` uses `ESMFold2Model.from_pretrained(..., load_esmc=False)`
  when `--require-lm-z-cache` or `--precomputed-lm-z-only` is set.
- If `--require-lm-z-cache` is set and a record has no cache entry, fail fast.
- If `--precomputed-lm-z-only` is set, also fail if `lm_z` is absent in any
  model forward call.

That first version skips loading the 24G ESM-C model during training. The small
ESMFold2 `language_model` projection can remain instantiated for compatibility.
If we really want to avoid even those pre-`lm_z` projection weights, add a later
constructor/load option that does not create `self.language_model` and refuses
live LM computation.

## Precompute Script

Add a separate script, for example:

```text
precompute_lm_z_cache.py
```

Workflow:

1. Load domain records from mdCATH.
2. Build ESMFold2 token features with `prepare_protein_features(sequence)`.
3. Load ESMFold2 with ESM-C:

```text
ESMFold2Model.from_pretrained(model, load_esmc=False)
model.load_esmc(esmc_model, precision=...)
model.eval()
```

4. For each unique sequence, call `model.compute_lm_z(...)` with `lm_mask_pct=0`.
5. Save unpadded `[L, L, 256]` tensor to `safetensors`.
6. Write/update manifest and index atomically.
7. Optionally recompute a random sample and assert tensor hashes match.

Batching by similar sequence length should be safe later, but the first script
can run one sequence at a time. Max local length is 200, so per-file size is
bounded at about 19.5 MiB for bf16.

## Deterministic Recompute

Required deterministic cache rules:

- Force `model.eval()`.
- Force `lm_mask_pct=0.0`; otherwise `compute_lm_hidden_states(...)` randomly
  masks LM tokens.
- Do not use `LanguageModelShim.forward(..., lm_dropout>0)`.
- Pin the cache dtype and ESM-C precision in the manifest.
- Prefer `fp32` for a canonical reference cache if byte-for-byte reproduction
  across environments matters more than speed/disk.
- Prefer `bf16` if the goal is to match the current CUDA Stage 1 training path
  with `--esmc-precision bf16`.
- Avoid `fp8` for the canonical cache until we have a hash-stable test for it.
- Set deterministic flags during precompute:

```python
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
```

For strict "no variance" semantics, define success as:

```text
same manifest inputs -> recomputed tensor bytes have same sha256
```

If the same cache must reproduce bitwise across different GPU models, CUDA
versions, or kernel libraries, use a slower canonical fp32/reference path and
validate it explicitly. Otherwise, bf16 GPU inference should be treated as
same-stack deterministic, with hash verification as the guardrail.

The existing per-loop `lm_z` dropout in `_run_one_loop(...)` is after cached
`lm_z`; it affects training stochasticity but not the cached tensor. Full
forward equivalence tests should reset the RNG before both live-LM and cached
forward calls, or temporarily set the LM-encoder dropout to zero.

## Validation Plan

1. Precompute one domain.
2. Load the cache and compare `model.compute_lm_z(...)` to the cached tensor
   with exact equality for the chosen dtype/precision stack.
3. Run one `forward_train(...)` call live with ESM-C and one with cached `lm_z`,
   resetting RNG before both calls. Loss/output tensors should match for the
   same stack.
4. Run a one-step training smoke test with `--require-lm-z-cache` and no ESM-C
   model path.
5. Run a missing-cache test and verify it fails before training starts.
6. Run a small validation/probe pass to confirm cached `lm_z` also reaches
   validation and denoise-probe paths.

## Implementation Order

1. Add model `lm_z` input plumbing and `compute_lm_z(...)`.
2. Add cache loader/padder to `train_md_esmfold2.py`.
3. Add `precompute_lm_z_cache.py`.
4. Add exact recompute/hash check.
5. Add `--require-lm-z-cache` training smoke test.
6. Optionally add the stricter `--precomputed-lm-z-only` model-load path that
   skips or deletes the LM producer modules.
