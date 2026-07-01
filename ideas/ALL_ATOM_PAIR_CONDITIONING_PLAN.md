# All-Atom Pair Conditioning Plan

## Motivation

The current MD-conditioning path reduces the current MD frame `x_t` to C-alpha
pair distances, then injects the resulting pair bias before the recurrent trunk.
For the `x_t -> x_t` identity diagnostic, that representation may be too lossy:
it removes sidechain state, most backbone detail, and all atom-level geometry
except C-alpha distances.

This plan tests a more informative but still trunk-compatible route: encode all
heavy atoms from `x_t` into the ESMFold2 pair tensor and inject it after the
stable recurrent update, but before the final two pair folding layers.

In the paper pseudocode, this is Algorithm 1 line 15:

```text
z <- PairFoldingLayers[2](Linear(z_T))
```

In the local implementation, this corresponds to:

```python
z = self.parcae_readout(z)
z = self.parcae_coda(z, pair_attention_mask=pair_mask)
```

The proposed insertion point is:

```python
z = self.parcae_readout(z)
z = z + self.late_all_atom_conditioning(...)
z = self.parcae_coda(z, pair_attention_mask=pair_mask)
```

## Key Idea

The trunk state `z` is token-pair shaped:

```text
z: [B, L, L, d_pair]
```

The MD frame is atom shaped:

```text
x_t: [B, A, 3]
atom_to_token: [B, A]
```

Therefore, all atoms must be compressed into one `d_pair` vector for each token
pair `(i, j)`. The simplest invariant design is to use all inter-token heavy-atom
distances:

```text
x_t atoms             [B, A, 3]
slotized atoms        [B, L, K, 3]
pair atom distances   [B, L, L, K, K]
RBF + atom identity   [B, L, L, K, K, h]
pool/project          [B, L, L, d_pair]
```

where `K` is the maximum number of atoms assigned to any token in the batch.

## Proposed Encoder

### 1. Slotize atoms by token

Build token-local atom tensors:

```text
atom_xyz_by_token:       [B, L, K, 3]
atom_mask_by_token:      [B, L, K]
atom_name_by_token:      [B, L, K, 4]
```

The source tensors are already available in `ESMFold2Model._forward_impl(...)`:

```text
x_t
atom_to_token
ref_atom_name_chars
atom_attention_mask
token_attention_mask
```

For protein-only mdCATH examples, `K` should be small enough for a first
implementation. A later implementation can use fixed residue atom slots if that
proves cleaner.

### 2. Compute all inter-token atom distances

For every token pair `(i, j)` and atom slots `(a, b)`:

```python
diff = atom_xyz_by_token[:, :, None, :, None, :] - atom_xyz_by_token[:, None, :, None, :, :]
dist = diff.norm(dim=-1)
```

Shape:

```text
dist: [B, L, L, K, K]
```

This is rotation- and translation-invariant, which avoids coordinate-frame
mismatch with the diffusion training augmentation.

### 3. Expand distances and atom identities

Use RBF features for distances:

```text
rbf(dist): [B, L, L, K, K, n_rbf]
```

Add atom identity information so the encoder can distinguish, for example,
`CB-CG` from `N-O` distances:

```text
atom_pair_features = [
    rbf(dist_ijab),
    embedding(atom_name_i_a),
    embedding(atom_name_j_b),
]
```

A simple first version can use atom-name character embeddings derived from
`ref_atom_name_chars`.

### 4. Atom-pair MLP and masked pooling

Apply a small shared MLP to every atom pair, then mask and pool over atom slots:

```text
h_ijab = MLP(atom_pair_features_ijab)
z_ij = masked_mean_ab(h_ijab)
```

Final output:

```text
z_xt: [B, L, L, d_pair]
```

The final projection should be zero-initialized so normal ESMFold2 behavior is
unchanged when training starts, and unchanged when `x_t` is absent.

### 5. Add timestep conditioning

Reuse the existing `FourierDTEncoder`:

```text
z_xt = z_all_atom + dt_encoder(dt)[:, None, None, :]
```

Then apply the token-pair mask:

```text
pair_mask = token_attention_mask[:, :, None] & token_attention_mask[:, None, :]
z_xt = z_xt * pair_mask[..., None]
```

## Optional Local Coordinate Features

All-atom distances encode rich geometry but do not directly provide chirality or
absolute orientation. A stronger version can add local-frame atom coordinates.

For each residue/token, build a backbone local frame from `N`, `CA`, and `C`, then
express that token's atom coordinates in the local frame:

```text
local_xyz_i_a: [B, L, K, 3]
```

Then add token-local summaries:

```text
s_i = Pool_a(MLP(atom_name_i_a, local_xyz_i_a))
z_ij += MLP(s_i + s_j)
```

This should help represent sidechain rotamers and local backbone geometry. It is
best treated as a second step after the distance-only encoder is verified.

## Why This Is Easier Than Early Injection

The existing MD-conditioning path is injected before the recurrent trunk loop.
The recurrent update can dilute or transform the conditioning signal over many
iterations.

Injecting after the recurrent loop and before `parcae_coda` gives the trainable
conditioning module a shorter path to the frozen distogram and diffusion heads:

```text
all-atom x_t -> z_xt -> 2 pair folding layers -> diffusion head -> x_pred
```

For the identity diagnostic, this should be easier to learn than early
conditioning.

## What This Can And Cannot Prove

This is still not a literal coordinate copy path. The trainable module only
modifies a pair representation, and the final coda plus diffusion head are fixed.
Therefore, exact all-atom `x_t -> x_t` reconstruction is not guaranteed.

Expected behavior:

- It should be much stronger than C-alpha-only pair conditioning.
- It should improve the identity diagnostic if the conditioning path is wired and
  trainable.
- It may recapitulate backbone and sidechain geometry only approximately.
- Failure to exactly copy `x_t` does not by itself prove the MD-conditioning
  plumbing is broken.

For a truly trivial copy test, the direct diffusion-module atom-conditioning plan
in `XT_ATOM_CONDITIONING_PLAN.md` is still the more direct route.

## Minimal Implementation Sketch

Add a new module near the existing `MDConditioning` classes:

```python
class AllAtomPairEncoder(nn.Module):
    def __init__(self, d_pair: int, n_rbf: int = 32, atom_emb_dim: int = 32):
        ...

    def forward(
        self,
        x_t: Tensor,
        atom_to_token: Tensor,
        ref_atom_name_chars: Tensor,
        atom_attention_mask: Tensor,
        token_attention_mask: Tensor,
    ) -> Tensor:
        ...
        return z_xt
```

Then create a late-conditioning wrapper:

```python
class LateAllAtomMDConditioning(nn.Module):
    def __init__(self, d_pair: int):
        self.atom_pair_encoder = AllAtomPairEncoder(d_pair)
        self.dt_encoder = FourierDTEncoder(d_pair)

    def forward(...):
        z_atom = self.atom_pair_encoder(...)
        z_dt = self.dt_encoder(dt)[:, None, None, :]
        return masked_pair(z_atom + z_dt)
```

Add it to `ESMFold2Model.__init__`:

```python
self.late_md_conditioning = LateAllAtomMDConditioning(d_pair=d_pair)
```

Inject it after `parcae_readout`:

```python
z = self.parcae_readout(z)
if x_t is not None and dt is not None:
    z = z + self.late_md_conditioning(
        x_t=x_t,
        dt=dt,
        atom_to_token=atom_to_token,
        ref_atom_name_chars=ref_atom_name_chars,
        atom_attention_mask=atm_mask,
        token_attention_mask=tok_mask,
    ).to(dtype=z.dtype)
z = self.parcae_coda(z, pair_attention_mask=pair_mask)
```

For the first experiment, disable the old early MD-conditioning addition so the
effect of the late all-atom path is isolated.

## Diagnostic Plan

Start with the identity target:

```text
--target-current-frame
```

Recommended sequence:

1. Tiny fixed-batch overfit run.
   The training loss should drop clearly if the late all-atom path can carry the
   target signal.

2. Fixed validation batches.
   The identity diagnostic should improve relative to the previous C-alpha-only
   flat baseline.

3. Compare against CA-only late injection.
   If all-atom late conditioning works and CA-only late conditioning does not,
   the bottleneck was representation loss, not only injection location.

4. Return to future-frame prediction.
   After `x_t -> x_t` works, test `x_t -> x_t+dt`.

## Success Criteria

Keep this path only if:

- Normal inference is unchanged when `x_t` and `dt` are absent.
- Only the new conditioning module is trainable in the diagnostic run.
- Gradients are present on the late conditioning module.
- Frozen ESMFold2 parameters remain frozen.
- Tiny fixed-batch identity training overfits.
- Fixed validation improves on the C-alpha-only identity diagnostic.

## Main Risks

- Memory scales as `B * L * L * K * K`; for long proteins this can be expensive.
- Distance-only features are invariant but may still be insufficient for exact
  all-atom reconstruction.
- If atom slotization is wrong, the model may silently learn corrupted atom-pair
  summaries.
- If both early and late conditioning are active, diagnostics become harder to
  interpret.

## Fallbacks

If memory is too high:

- Use only heavy atoms from the backbone plus `CB` as an intermediate test.
- Pool atom features per token first, then build pair features from pooled token
  summaries.
- Randomly subsample atom pairs during training.
- Use fixed residue atom slots and avoid dynamic per-batch `K` when possible.

If identity learning still fails:

- Check masks and atom-to-token slotization.
- Verify nonzero gradients on the late conditioning module.
- Run an overfit test with `parcae_coda` unfrozen to determine whether the frozen
  coda is the bottleneck.
- Switch to direct atom conditioning in the diffusion module, where `x_t` can be
  used as an atom-level coordinate input.
