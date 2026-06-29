#!/usr/bin/env python3
"""Small mdCATH training loop for an MD-conditioned ESMFold2.

This script deliberately lives outside the `transformers/` submodule.  It
assumes the model will expose a grad-enabled method with this contract:

    model.forward_train(..., x_t=x_t, dt=dt,
                        target_atom_coords=x_target,
                        target_atom_mask=atom_mask)

`x_t` is the current MD frame.  `dt` is the MD timestep in seconds.  Both are
plain tensors before the model decides how to turn them into extra z_lm
conditioning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
LOCAL_TRANSFORMERS = ROOT / "transformers" / "src"
sys.path.insert(0, str(LOCAL_TRANSFORMERS))

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402
from transformers.models.esmfold2.protein_utils import (  # noqa: E402
    PROTEIN_1TO3,
    PROTEIN_HEAVY_ATOMS,
    prepare_protein_features,
)


_H5PY = None
_NP = None

RESIDUE_NAME_ALIASES = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
}

ATOM_NAME_ALIASES = {
    ("ILE", "CD1"): ("CD",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="biohub/ESMFold2")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mdcath_320K_len_le200" / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--temperature", default="320")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means use all smart batches")
    parser.add_argument("--length-max", type=int, default=1500)
    parser.add_argument("--batch-exp", type=float, default=1.9)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-delta-frames", type=int, default=100)
    parser.add_argument("--frame-time-ns", type=float, default=1.0)
    parser.add_argument("--geometric-p", type=float, default=0.1)
    parser.add_argument("--num-sampling-steps", type=int, default=1)
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def need_h5py():
    global _H5PY, _NP
    if _H5PY is None or _NP is None:
        try:
            import h5py
            import numpy as np
        except ImportError as err:
            raise RuntimeError("Install h5py and numpy before running: python3 -m pip install h5py numpy") from err
        _H5PY, _NP = h5py, np
    return _H5PY, _NP


def as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return as_text(value.tolist())
    if isinstance(value, list):
        return "".join(as_text(v) for v in value)
    return str(value)


def domain_from_path(path: Path) -> str:
    name = path.stem
    prefix = "mdcath_dataset_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def load_domain_records(data_dir: Path, temperature: str) -> list[dict]:
    h5py, _ = need_h5py()
    records = []
    skipped_unmappable = 0

    for path in sorted(data_dir.glob("mdcath_dataset_*.h5")):
        domain = domain_from_path(path)
        with h5py.File(path, "r") as h5:
            group = h5[domain]
            sequence = as_text(group["sequence"][()])
            pdb_text = as_text(group["pdbProteinAtoms"][()])
            protein_groups = [
                g
                for g in pdb_residue_groups(pdb_text)
                if g["resname"] in PROTEIN_HEAVY_ATOMS
            ]
            if len(protein_groups) != len(sequence):
                skipped_unmappable += 1
                continue
            available_temperatures = [k for k in group.keys() if k.isdigit()]
            if temperature not in available_temperatures and not available_temperatures:
                continue

            cluster = as_text(group.attrs.get("cluster", domain))
            records.append(
                {
                    "path": path,
                    "domain": domain,
                    "cluster": cluster,
                    "length": len(sequence),
                    "sequence": sequence,
                }
            )

    if not records:
        raise FileNotFoundError(f"no mdCATH HDF5 files found under {data_dir}")
    if skipped_unmappable:
        print(f"skipped_unmappable_domains={skipped_unmappable}")
    return records


def length_to_batch(length: int, length_max: int, batch_exp: float) -> int:
    return math.floor((float(length_max) / float(length)) ** float(batch_exp))


def one_record_per_cluster(records: list[dict]) -> list[dict]:
    by_cluster: dict[str, list[dict]] = {}
    for record in records:
        by_cluster.setdefault(record["cluster"], []).append(record)
    return [random.choice(group) for group in by_cluster.values()]


def make_smart_batches(records: list[dict], length_max: int, batch_exp: float) -> list[list[dict]]:
    sampled = one_record_per_cluster(records)
    sampled = [r for r in sampled if length_to_batch(r["length"], length_max, batch_exp) > 0]
    sampled.sort(key=lambda r: r["length"] + 2.0 * random.gauss(0.0, 1.0))

    batches: list[list[dict]] = []
    current: list[dict] = []
    current_max_len = 0

    for record in sampled:
        potential_max_len = max(current_max_len, record["length"])
        allowed = length_to_batch(potential_max_len, length_max, batch_exp)

        if not current or len(current) + 1 <= allowed:
            current.append(record)
            current_max_len = potential_max_len
        else:
            batches.append(current)
            current = [record]
            current_max_len = record["length"]

    if current:
        batches.append(current)

    random.shuffle(batches)
    return batches


def find_axis(shape: tuple[int, ...], size: int, name: str) -> int:
    axes = [axis for axis, axis_size in enumerate(shape) if axis_size == size]
    if len(axes) != 1:
        raise ValueError(f"could not identify {name} axis in shape {shape}")
    return axes[0]


def truncated_geometric(max_value: int, p: float) -> int:
    u = random.random()
    tail = 1.0 - (1.0 - p) ** max_value
    return max(1, min(max_value, math.ceil(math.log(1.0 - u * tail) / math.log(1.0 - p))))


def choose_delta(n_frames: int, max_delta_frames: int, geometric_p: float) -> int:
    max_delta = min(max_delta_frames, n_frames - 1)
    if random.random() < 0.5:
        return truncated_geometric(max_delta, geometric_p)
    return random.randint(1, max_delta)


def read_two_frames(coords, num_atoms: int, num_frames: int, frame_a: int, frame_b: int):
    _, np = need_h5py()
    xyz_axis = find_axis(coords.shape, 3, "xyz")
    atom_axis = find_axis(coords.shape, num_atoms, "atom")
    frame_axis = find_axis(coords.shape, num_frames, "frame")

    frames = np.array([frame_a, frame_b], dtype=int)
    sorted_frames, restore_order = np.unique(frames, return_inverse=True)

    selection = [slice(None)] * coords.ndim
    selection[frame_axis] = sorted_frames
    data = coords[tuple(selection)]
    data = np.take(data, restore_order, axis=frame_axis)
    data = np.moveaxis(data, (frame_axis, atom_axis, xyz_axis), (0, 1, 2))
    return data.astype("float32", copy=False)


def atom_template_lines(pdb_text: str) -> list[str]:
    return [
        line
        for line in pdb_text.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


def pdb_atom_name(line: str) -> str:
    return line[12:16].strip()


def pdb_residue_name(line: str) -> str:
    name = line[17:20].strip()
    return RESIDUE_NAME_ALIASES.get(name, name)


def pdb_residue_key(line: str) -> tuple[str, str, str]:
    return (line[21].strip(), line[22:26].strip(), line[26].strip())


def pdb_residue_groups(pdb_text: str) -> list[dict]:
    groups: list[dict] = []
    current_key: tuple[str, str, str] | None = None
    current: dict | None = None
    for md_atom_idx, line in enumerate(atom_template_lines(pdb_text)):
        key = pdb_residue_key(line)
        if key != current_key:
            current = {
                "key": key,
                "resname": pdb_residue_name(line),
                "atoms": [],
            }
            groups.append(current)
            current_key = key
        assert current is not None
        current["atoms"].append((pdb_atom_name(line), md_atom_idx))
    return groups


def feature_to_md_atom_indices(sequence: str, pdb_text: str, atom_count: int):
    _, np = need_h5py()
    groups = pdb_residue_groups(pdb_text)
    groups = [g for g in groups if g["resname"] in PROTEIN_HEAVY_ATOMS]
    expected_resnames = [PROTEIN_1TO3.get(letter, "UNK") for letter in sequence]
    if len(groups) != len(expected_resnames):
        raise ValueError(
            f"PDB template has {len(groups)} protein residue groups, "
            f"but sequence has {len(expected_resnames)} residues"
        )

    indices = np.full(atom_count, -1, dtype=np.int64)
    feature_atom_idx = 0
    for token_idx, (res3, group) in enumerate(zip(expected_resnames, groups)):
        if res3 != "UNK" and group["resname"] != res3:
            raise ValueError(
                f"Residue {token_idx} is {res3} in sequence but "
                f"{group['resname']} in PDB atom template"
            )
        by_name = {name: md_idx for name, md_idx in group["atoms"]}
        for atom_name in PROTEIN_HEAVY_ATOMS[res3]:
            md_atom_idx = by_name.get(atom_name)
            if md_atom_idx is None:
                for alias in ATOM_NAME_ALIASES.get((res3, atom_name), ()):
                    md_atom_idx = by_name.get(alias)
                    if md_atom_idx is not None:
                        break
            if md_atom_idx is None:
                raise ValueError(
                    f"Missing atom {atom_name} for residue {token_idx} {res3} "
                    "in PDB atom template"
                )
            if feature_atom_idx >= atom_count:
                raise ValueError(
                    f"ESMFold2 feature atom index {feature_atom_idx} exceeds "
                    f"atom_count {atom_count}"
                )
            indices[feature_atom_idx] = md_atom_idx
            feature_atom_idx += 1
    return indices


def map_md_coords_to_features(coords, feature_to_md_atom):
    _, np = need_h5py()
    mapped = np.zeros((feature_to_md_atom.shape[0], 3), dtype=np.float32)
    valid = feature_to_md_atom >= 0
    mapped[valid] = coords[feature_to_md_atom[valid]]
    return mapped, valid


def center_masked(coords, mask):
    out = coords.copy()
    out[~mask] = 0
    out[mask] -= out[mask].mean(axis=0, keepdims=True)
    return out


def kabsch_align(moving, target, mask=None):
    _, np = need_h5py()
    moving_fit = moving if mask is None else moving[mask]
    target_fit = target if mask is None else target[mask]
    cov = moving_fit.T @ target_fit
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    return moving @ rot


def pad_atom_coords(coords, atom_count: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(1, atom_count, 3, dtype=torch.float32, device=device)
    n_atoms = coords.shape[0]
    if n_atoms > atom_count:
        raise ValueError(f"HDF5 has {n_atoms} atoms, but ESMFold2 features have {atom_count}")
    out[0, :n_atoms] = torch.from_numpy(coords).to(device=device)
    return out


def tensorize_record(record: dict, args: argparse.Namespace, feature_cache: dict, device: torch.device):
    h5py, _ = need_h5py()
    path = record["path"]
    domain = record["domain"]

    with h5py.File(path, "r") as h5:
        group = h5[domain]
        sequence = record["sequence"]
        temperature = args.temperature if args.temperature in group else random.choice([k for k in group if k.isdigit()])
        run = random.choice(sorted(group[temperature].keys(), key=int))
        run_group = group[temperature][run]

        coords = run_group["coords"]
        num_atoms = int(group.attrs["numProteinAtoms"])
        num_frames = int(run_group.attrs.get("numFrames", coords.shape[-1]))
        pdb_text = as_text(group["pdbProteinAtoms"][()])

        delta_frames = choose_delta(num_frames, args.max_delta_frames, args.geometric_p)
        frame = random.randint(0, num_frames - delta_frames - 1)
        pair = read_two_frames(coords, num_atoms, num_frames, frame, frame + delta_frames)

    if sequence not in feature_cache:
        feature_cache[sequence] = prepare_protein_features(sequence)

    features = {k: v.to(device) for k, v in feature_cache[sequence].items()}
    atom_count = features["ref_pos"].shape[1]
    feature_to_md_atom = feature_to_md_atom_indices(sequence, pdb_text, atom_count)

    current, valid_atoms = map_md_coords_to_features(pair[0], feature_to_md_atom)
    target, _ = map_md_coords_to_features(pair[1], feature_to_md_atom)
    current = center_masked(current, valid_atoms)
    target = kabsch_align(center_masked(target, valid_atoms), current, valid_atoms)
    x_t = pad_atom_coords(current, atom_count, device)
    x_target = pad_atom_coords(target, atom_count, device)

    atom_mask = torch.from_numpy(valid_atoms).to(device=device).unsqueeze(0)
    atom_mask &= features["atom_attention_mask"]

    dt_seconds = delta_frames * args.frame_time_ns * 1e-9
    dt = torch.tensor([[dt_seconds]], dtype=torch.float32, device=device)

    extra = {
        "x_t": x_t,
        "dt": dt,
        "target_atom_coords": x_target,
        "target_atom_mask": atom_mask,
    }
    info = {
        "domain": domain,
        "temperature": temperature,
        "run": run,
        "frame": frame,
        "delta_frames": delta_frames,
        "dt_seconds": dt_seconds,
    }
    return features, extra, info


def pad_like(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    out = torch.zeros(shape, dtype=x.dtype, device=x.device)
    slices = tuple(slice(0, n) for n in x.shape)
    out[slices] = x
    return out


def pad_features(feature_list: list[dict]) -> dict:
    max_l = max(f["res_type"].shape[1] for f in feature_list)
    max_a = max(f["ref_pos"].shape[1] for f in feature_list)

    token_keys = {
        "token_index",
        "residue_index",
        "asym_id",
        "sym_id",
        "entity_id",
        "mol_type",
        "res_type",
        "input_ids",
        "token_attention_mask",
        "deletion_mean",
        "distogram_atom_idx",
    }
    atom_keys = {
        "ref_pos",
        "ref_element",
        "ref_charge",
        "ref_atom_name_chars",
        "ref_space_uid",
        "atom_attention_mask",
        "atom_to_token",
    }
    msa_keys = {"msa", "msa_attention_mask", "has_deletion", "deletion_value"}

    batched = {}
    for key in feature_list[0].keys():
        padded = []
        for f in feature_list:
            x = f[key]
            if key in token_keys:
                shape = (1, max_l)
            elif key == "token_bonds":
                shape = (1, max_l, max_l, 1)
            elif key in msa_keys:
                shape = (1, x.shape[1], max_l)
            elif key == "ref_pos":
                shape = (1, max_a, 3)
            elif key == "ref_atom_name_chars":
                shape = (1, max_a, 4)
            elif key in atom_keys:
                shape = (1, max_a)
            else:
                raise KeyError(f"do not know how to pad feature {key}")
            padded.append(pad_like(x, shape))
        batched[key] = torch.cat(padded, dim=0)
    return batched


def pad_extras(extra_list: list[dict]) -> dict:
    max_a = max(e["x_t"].shape[1] for e in extra_list)
    x_t = torch.cat([pad_like(e["x_t"], (1, max_a, 3)) for e in extra_list], dim=0)
    target = torch.cat([pad_like(e["target_atom_coords"], (1, max_a, 3)) for e in extra_list], dim=0)
    mask = torch.cat([pad_like(e["target_atom_mask"], (1, max_a)) for e in extra_list], dim=0)
    dt = torch.cat([e["dt"] for e in extra_list], dim=0)
    return {
        "x_t": x_t,
        "dt": dt,
        "target_atom_coords": target,
        "target_atom_mask": mask,
    }


def tensorize_batch(records: list[dict], args: argparse.Namespace, feature_cache: dict, device: torch.device):
    feature_list = []
    extra_list = []
    infos = []

    for record in records:
        features, extra, info = tensorize_record(record, args, feature_cache, device)
        feature_list.append(features)
        extra_list.append(extra)
        infos.append(info)

    features = pad_features(feature_list)
    extra = pad_extras(extra_list)
    info = {
        "batch_size": len(records),
        "max_length": max(r["length"] for r in records),
        "domains": ";".join(i["domain"] for i in infos),
        "mean_dt_seconds": sum(i["dt_seconds"] for i in infos) / len(infos),
    }
    return features, extra, info


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 4:
        pred = pred[:, 0]
    sq = (pred - target).square().sum(dim=-1)
    return (sq * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def training_loss(
    model, features: dict, extra: dict, args: argparse.Namespace
) -> tuple[torch.Tensor, dict]:
    if not hasattr(model, "forward_train"):
        raise RuntimeError(
            "Expected model.forward_train(..., x_t, dt, target_atom_coords, target_atom_mask). "
            "Add that method to ESMFold2Model before running this script."
        )

    out = model.forward_train(
        **features,
        **extra,
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
    )
    if "loss" in out:
        return out["loss"], out

    coord_key = "sample_atom_coords" if "sample_atom_coords" in out else "x_pred"
    loss = masked_mse(out[coord_key], extra["target_atom_coords"], extra["target_atom_mask"])
    return loss, out


def save_checkpoint(path: Path, model, optimizer, args: argparse.Namespace, epoch: int, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def append_loss_row(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)

    records = load_domain_records(args.data_dir, args.temperature)

    run_name = "md_esmfold2_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    model = ESMFold2Model.from_pretrained(args.model)
    model.to(device)
    model.train()
    if hasattr(model, "set_kernel_backend"):
        model.set_kernel_backend(None)

    if not hasattr(model, "md_conditioning"):
        raise RuntimeError(
            "Expected ESMFold2Model.md_conditioning for Stage 1 MD fine-tuning"
        )
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.md_conditioning.parameters():
        param.requires_grad_(True)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after enabling md_conditioning")
    trainable_count = sum(param.numel() for param in trainable_params)
    total_count = sum(param.numel() for param in model.parameters())
    print(
        f"trainable_params={trainable_count} total_params={total_count} "
        "trainable_module=md_conditioning"
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    feature_cache: dict = {}
    loss_csv = run_dir / "loss.csv"

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        batches = make_smart_batches(records, args.length_max, args.batch_exp)
        if args.steps_per_epoch > 0:
            batches = batches[: args.steps_per_epoch]

        sample_count = sum(len(batch) for batch in batches)
        print(f"epoch={epoch} batches={len(batches)} samples={sample_count}")

        for batch_records in batches:
            optimizer.zero_grad(set_to_none=True)
            features, extra, info = tensorize_batch(batch_records, args, feature_cache, device)
            loss, _ = training_loss(model, features, extra, args)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            row = {
                "epoch": epoch,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                **info,
            }
            append_loss_row(loss_csv, row)

            if global_step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"step_{global_step:07d}.pt", model, optimizer, args, epoch, global_step)
                print(f"epoch={epoch} step={global_step} loss={row['loss']:.5f}")

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, args, args.epochs, global_step)
    print(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
