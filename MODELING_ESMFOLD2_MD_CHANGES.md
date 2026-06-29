# ESMFold2 MD Changes

This file tracks local changes made to:

`/home/theodor/MD_ESMFold2/transformers/src/transformers/models/esmfold2/modeling_esmfold2.py`

The grad-enabled training path also required one supporting change in:

`/home/theodor/MD_ESMFold2/transformers/src/transformers/models/esmfold2/modeling_esmfold2_common.py`

Compare against the upstream baseline from the nested `transformers` repo with:

```bash
cd /home/theodor/MD_ESMFold2/transformers
git diff main -- src/transformers/models/esmfold2/modeling_esmfold2.py
```

Current comparison context:

- Nested repo branch: `MD`
- Baseline branch: `main`
- Last reviewed: 2026-06-29
- Current `modeling_esmfold2.py` diff size: 414 insertions, 21 deletions
- Current `modeling_esmfold2_common.py` diff size: 11 insertions, 3 deletions

## Purpose

The local changes add an MD-conditioning path so `train_md_esmfold2.py` can pass a current MD frame, `x_t`, and a timestep, `dt`, into ESMFold2. The model turns those tensors into an additive pair-state bias before the folding trunk runs.

The normal ESMFold2 inference behavior is intended to remain unchanged when `x_t` and `dt` are not provided.

## Added Components

### `FourierDTEncoder`

Encodes the MD time delta `dt` into pair-channel features.

- Accepts `dt` in seconds.
- Converts seconds to nanoseconds.
- Uses `log(dt_ns)` for numerical scaling.
- Adds sinusoidal Fourier features over log-time.
- Adds a scaled linear `dt_ns / max_dt_ns` feature.
- Projects the resulting feature vector to `d_pair`.
- Zero-initializes the final linear projection so the module starts as a no-op.

### `CAFramePairEncoder`

Encodes the current MD frame geometry as token-pair features.

- Takes C-alpha coordinates shaped like `[batch, tokens, 3]`.
- Computes pairwise C-alpha distances with `torch.cdist`.
- Expands distances with radial basis functions over 2 to 40 Angstrom.
- Projects RBF features to `d_pair`.
- Applies the token-pair attention mask.
- Zero-initializes the final projection so the module starts as a no-op.

### `MDConditioning`

Combines frame and timestep conditioning into an `lm_z`-shaped pair bias.

- Finds C-alpha atoms by matching `ref_atom_name_chars` against the encoded atom name `CA`.
- Uses `atom_to_token` to scatter C-alpha coordinates from atom space into token space.
- Validates that `x_t` has shape `[B, A, 3]` and matches `atom_to_token`.
- Raises an error if any valid token lacks a C-alpha atom.
- Adds the C-alpha pair encoding and timestep encoding.
- Applies the token-pair mask before returning the pair bias.

## Integration Into `ESMFold2Model`

The model constructor now creates:

```python
self.md_conditioning = MDConditioning(d_pair=d_pair)
```

After `self.post_init()`, the MD-conditioning outputs are zero-initialized again:

```python
self.md_conditioning.zero_init_output()
```

This is meant to preserve pretrained behavior until MD conditioning learns a useful signal.

## Forward API Changes

The original inference path has been split into:

- `_forward_impl(...)`: shared implementation.
- `forward(...)`: inference wrapper that preserves `@torch.inference_mode()`.
- `forward_train(...)`: grad-enabled wrapper for MD training.

`ESMFold2Model.forward(...)` and `ESMFold2Model.forward_train(...)` accept two optional MD-conditioning keyword arguments:

```python
x_t: Tensor | None = None
dt: Tensor | None = None
```

The model requires them to be provided together:

```python
if (x_t is None) != (dt is None):
    raise ValueError("x_t and dt must be provided together")
```

When both are present, the model computes:

```python
md_z = self.md_conditioning(...)
```

and injects it through the existing language-model pair-conditioning path:

```python
lm_z = md_z if lm_z is None else lm_z + md_z.to(dtype=lm_z.dtype)
```

This means MD conditioning is added before the recycling/trunk loop, using the same pair-state injection mechanism as `lm_z`.

## Training Path

`ESMFold2Model.forward_train(...)` accepts:

```python
x_t: Tensor | None = None
dt: Tensor | None = None
target_atom_coords: Tensor | None = None
target_atom_mask: Tensor | None = None
```

It calls `_forward_impl(...)` with:

- `train_structure=True`
- `compute_distogram=False`
- `compute_confidence=False`
- `num_diffusion_samples=1` by default

When target coordinates and a target mask are provided, it computes masked coordinate MSE and returns it as `output["loss"]`.

The inference-only structure sampler in `DiffusionStructureHead` was split into:

- `_sample_impl(...)`: shared sampling implementation.
- `sample(...)`: inference wrapper that preserves `@torch.inference_mode()`.
- `sample_train(...)`: grad-enabled sampler used by `forward_train(...)`.

## Relationship To `train_md_esmfold2.py`

The training script currently prepares these tensors:

- `x_t`: centered current MD frame, padded to the ESMFold2 atom count.
- `dt`: frame delta converted to seconds.
- `target_atom_coords`: centered and Kabsch-aligned future frame, padded to atom count.
- `target_atom_mask`: valid atom mask intersected with ESMFold2 atom attention mask.

The model changes now support the training script call:

```python
model.forward_train(**features, **extra)
```

`forward_train(...)` returns `output["loss"]` when the script provides `target_atom_coords` and `target_atom_mask`.

## Stage 1 Fine-Tuning

`train_md_esmfold2.py` now freezes the whole model, re-enables gradients only for `model.md_conditioning`, and builds the optimizer from trainable parameters only:

- Set all model parameters to `requires_grad_(False)`.
- Set `model.md_conditioning.parameters()` to `requires_grad_(True)`.
- Build the optimizer from trainable parameters only.

## Known Remaining Gap

Optional later optimization: precompute and cache `lm_hidden_states` per sequence so MD training does not repeatedly run frozen ESMC.
