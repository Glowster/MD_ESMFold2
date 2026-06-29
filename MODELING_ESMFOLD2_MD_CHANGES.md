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
- Current `modeling_esmfold2.py` diff size: 436 insertions, 44 deletions
- Current `modeling_esmfold2_common.py` diff size: 113 insertions, 3 deletions

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

When target coordinates and a target mask are provided, it calls the diffusion head's single-noise-level training path rather than the inference sampler.

`DiffusionStructureHead.train_denoising(...)` now owns the diffusion training loss:

- Samples `sigma` from the configured ESMFold2 training noise distribution, using `sigma_data`, `train_noise_log_mean`, and `train_noise_log_std`.
- Applies the same center/random-rotation/random-translation augmentation used by the structure head.
- Constructs `x_noisy = x_target + sigma * noise`.
- Runs `diffusion_module(...)` once.
- Computes the masked MSE between `x_denoised` and the augmented target.
- Returns `output["loss"]`, `x_pred`, `x_noisy`, `noise_sigma`, `denoise_rmsd`, `noisy_rmsd`, and `noise_sigma_mean`.

This replaced an earlier incorrect debug path where `forward_train(...)` called the inference sampler with `num_sampling_steps=1` and compared `sample_atom_coords` directly to the target. That path produced losses around 50,000 because it was still at inference-noise scale and was not a diffusion training objective.

The inference-only structure sampler in `DiffusionStructureHead` was split into:

- `_sample_impl(...)`: shared sampling implementation.
- `sample(...)`: inference wrapper that preserves `@torch.inference_mode()`.
- `sample_train(...)`: grad-enabled sampler for explicit debug experiments.
- `sample_train(...)` is retained for explicit debug use, but `forward_train(...)` with targets no longer uses it for the loss.

## Relationship To `train_md_esmfold2.py`

The training script currently prepares these tensors:

- `x_t`: current MD frame remapped from mdCATH PDB atom order into ESMFold2 heavy-atom feature order, centered over valid mapped atoms, and padded to the ESMFold2 atom count.
- `dt`: frame delta converted to seconds.
- `target_atom_coords`: future frame remapped into ESMFold2 heavy-atom order, centered, Kabsch-aligned to `x_t`, and padded to atom count.
- `target_atom_mask`: valid mapped heavy-atom mask intersected with ESMFold2 atom attention mask.

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
- Default MD training sampling uses `--num-sampling-steps 1`, with `--num-loops` also exposed for smoke/debug runs.
- `loss.csv` is appended once per optimizer step, so each batch contributes one row for plotting the training loss curve.
- `loss.csv` now also logs `denoise_rmsd`, `noisy_rmsd`, and `noise_sigma_mean` when the model returns them.
- The script accepts `--esmc-model` and `--esmc-precision`, allowing fully local/offline loading of ESMFold2 plus the ESMC backbone.
- Checkpoints are lightweight by default: they save `md_conditioning`, optimizer state, trainable parameter names, args, epoch, and step. The frozen ESMFold2/ESMC weights are not saved unless `--save-full-checkpoint` is explicitly passed.
- Step checkpoints are controlled by `--save-every`; epoch checkpoints are controlled by `--save-every-epochs`, which defaults to 10. `last.pt` is always saved at the end.
- Checkpoint args are saved in JSON-serializable form so PyTorch 2.6 can load the checkpoint with its default `weights_only=True` behavior.

The script also now builds an atom map from each HDF5 file's `pdbProteinAtoms` template, because mdCATH atom order includes hydrogens and terminal/capping atoms and does not match ESMFold2 heavy-atom feature order directly.

Atom mapping normalizes common histidine protonation residue names (`HID`, `HIE`, `HIP`, `HSD`, `HSE`, `HSP`) to `HIS`, and accepts mdCATH's `ILE CD` name as ESMFold2's `ILE CD1`.

Full atom-map validation over `data/mdcath_320K_len_le200/data` checked 4,470 usable files with 3,817,536 mapped heavy atoms and zero mapping failures. The loader skips 12 files where `pdbProteinAtoms` has a different number of protein residue groups than the HDF5 `sequence`.

## Validation Status

- A tiny random-model runtime smoke test passed: `forward_train(...)` produced a grad-enabled loss, `loss.backward()` produced gradients on `md_conditioning`, and no non-`md_conditioning` parameters had gradients.
- Full local model snapshots were downloaded under `/home/theodor/MD_ESMFold2/hf_models`:
  - `biohub_ESMFold2`: 1.3G
  - `biohub_ESMC-6B`: 24G, with all six safetensor shards present
- A full pretrained one-step smoke test passed using the downloaded local snapshots, one mdCATH domain, `num_loops=1`, and `num_sampling_steps=1`.
  - Device used: CPU, because this environment's PyTorch build could not use the installed NVIDIA driver.
  - Selected domain: `1a92A00`, length 50.
  - Trainable parameters: 149,124, all from `md_conditioning`.
  - Total loaded parameters: 6,580,251,553.
  - Loss before step: 50,470.132812.
  - `loss.requires_grad=True`.
  - `md_conditioning` gradient parameters: 12.
  - Sum of absolute `md_conditioning` gradients: 532.71000358.
  - Non-`md_conditioning` gradient parameters: 0.
  - `optimizer.step()` completed.
- A bounded Stage 1 GPU run was started in tmux session `md_esmfold2_gpu0_stage1` on GPU 0 with:
  - `torch==2.6.0+cu124`, matching the NVIDIA 550.144.03 driver.
  - `CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`.
  - Local offline model paths for ESMFold2 and ESMC-6B.
  - `--esmc-precision bf16`.
  - `--epochs 20`, `--steps-per-epoch 10`, `--length-max 300`.
  - `--num-loops 1`, `--num-sampling-steps 1`.
  - `--save-every-epochs 10`, `--save-every 0`.
  - Run directory: `/home/theodor/MD_ESMFold2/runs/md_esmfold2_20260629_155527`.
  - Log file: `/home/theodor/MD_ESMFold2/runs/tmux_logs/md_esmfold2_gpu0_stage1.log`.
  - This run was stopped because it used the incorrect inference-sampler training objective.
- After the diffusion-training fix, a corrected full-model GPU smoke test passed on GPU 0:
  - Loss: 102.74905395507812.
  - `denoise_rmsd`: 10.136520385742188.
  - `noisy_rmsd`: 76.03369903564453.
  - `noise_sigma_mean`: 43.316070556640625.
  - `md_conditioning` received gradients; non-`md_conditioning` parameters had no gradients.
- A corrected one-step `train_md_esmfold2.py` run passed on GPU 0:
  - Run directory: `/home/theodor/MD_ESMFold2/runs/md_esmfold2_20260629_162253`.
  - `loss.csv` contains `loss`, `denoise_rmsd`, `noisy_rmsd`, and `noise_sigma_mean`.
  - `checkpoints/last.pt` is about 1.8 MB, contains `md_conditioning` only, and does not contain full model weights.
  - The checkpoint loads with PyTorch 2.6 default `torch.load(..., weights_only=True)`.

## Known Remaining Gap

- The CPU full-model smoke test needed ESMC loaded with `precision="fp32"`. Loading ESMC as `bf16` on CPU produced a dtype mismatch between bf16 LM hidden states and fp32 ESMFold2 projection weights.
- Optional later optimization: precompute and cache `lm_hidden_states` per sequence so MD training does not repeatedly run frozen ESMC.
