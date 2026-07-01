# Current-Frame Atom Conditioning Plan

## Motivation

The `md_esmfold2_20260630_120026` diagnostic trained with `target_atom_coords = x_t`
instead of `x_t + dt`, but validation did not improve. That is a strong negative
signal for the current Stage 1 MD-conditioning path, where `x_t` is reduced to
C-alpha pair distances and injected as a pair-state bias before the folding trunk.

The next experiment should give the structure denoiser a more direct atom-level
route from the current MD frame to the predicted coordinates.

## Key Constraint

Do not simply concatenate raw `x_t` onto the existing atom feature vector:

```text
f_atom = [ref_pos, charge, mask, element, atom_name, raw_x_t]
```

Raw `x_t` is in an arbitrary global coordinate frame. The existing `ref_pos` values
are canonical local residue-template coordinates, not folded-structure coordinates.
Naively mixing those coordinate systems would make the conditioning rotation- and
translation-dependent in a way the model is not designed to handle.

## Proposed Direction

Add current-frame atom conditioning directly to the diffusion atom encoder path,
while keeping the folding trunk unchanged for the first experiment.

Conceptually:

```text
DiffusionModule(x_noisy, t_hat, f_atom, f_inputs, z_trunk, x_t)
```

Inside the diffusion module, build atom-query conditioning from:

```text
r_noisy
r_xt
r_noisy - r_xt
```

where `r_xt` is the current-frame atom coordinate tensor transformed into the same
training/inference coordinate frame as the denoising target/noisy sample and scaled
with a compatible coordinate normalization.

## Required Frame Handling

During training, `train_denoising(...)` applies center/random-rotation/random-translation
augmentation to `target_atom_coords` before constructing `x_noisy`. If `x_t` is used
as a coordinate input, it must receive the same rigid transform as the target.

Required behavior:

- The target frame and `x_t` must be centered/augmented together.
- `x_noisy` must be sampled from the augmented target.
- The model must see `x_t_augmented`, not the unaugmented original `x_t`.
- Masks for `x_t` should follow `target_atom_mask & ref_mask`.

This keeps `x_t`, `x_noisy`, and the target in a consistent coordinate frame.

## Minimal Implementation Sketch

1. Extend the training API so `forward_train(...)` accepts optional `x_t_atom_coords`
   or reuses `x_t` for atom-level coordinates.

2. Thread the tensor through:

```text
ESMFold2Model.forward_train(...)
DiffusionStructureHead.train_denoising(...)
DiffusionModule.forward(...)
ESMFold2AtomEncoder.forward(...)
```

3. In `train_denoising(...)`, augment `target_atom_coords` and `x_t` with the same
   random transform.

4. In `DiffusionModule.forward(...)`, normalize both coordinate inputs on the same
   scale. A first conservative choice:

```text
r_noisy = x_noisy / sqrt(t_hat^2 + sigma_data^2)
r_xt    = x_t_aug / sqrt(t_hat^2 + sigma_data^2)
```

5. In the structure-prediction atom encoder, replace the current coordinate linear
   input:

```text
Linear([r_noisy, pred_r1])
```

with a slightly wider coordinate conditioning input:

```text
Linear([r_noisy, r_xt, r_noisy - r_xt, pred_r1])
```

or, for the smallest first patch:

```text
Linear([r_noisy, r_xt, r_noisy - r_xt])
```

6. Zero-initialize the new coordinate-conditioning projection if preserving pretrained
   behavior is important. If the projection replaces an existing pretrained
   `coords_linear`, preserve the old `r_noisy` weights and initialize only the new
   `r_xt` channels to zero.

## Diagnostic Plan

Start with the identity diagnostic before returning to future-frame prediction:

```text
--target-current-frame
```

Recommended test sequence:

1. Tiny fixed-batch overfit run.
   The model should rapidly reduce loss when target is exactly `x_t`.

2. Fixed validation batches.
   The `x_t -> x_t` validation loss should move clearly below the previous flat
   baseline around loss `43.21`, denoise RMSD `6.57 A`.

3. Future-frame run.
   Only after the identity diagnostic works, switch back to `x_t -> x_t+dt`.

## Success Criteria

The new path is worth keeping only if:

- The model can overfit `x_t -> x_t` on a tiny fixed subset.
- Fixed validation improves on the identity diagnostic.
- The implementation does not change normal ESMFold2 inference when `x_t` is absent.
- Missing or partial atom mappings are masked rather than silently corrupting atoms.

## Interpretation

If direct atom-level `x_t` conditioning cannot learn the identity diagnostic, the
problem is likely not just the representation choice. It would point instead to
training wiring, augmentation mismatch, optimization, masking, or the denoising
objective.

If the identity diagnostic works but future-frame prediction does not, then the
conditioning path is functional, but the remaining problem is learning physical
dynamics from the available MD data and timestep signal.
