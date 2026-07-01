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
    OUTPUT_TO_PDB_FEATURE_KEYS,
    PROTEIN_1TO3,
    PROTEIN_HEAVY_ATOMS,
    output_to_pdb,
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

DEFAULT_FORCE_VAL_DOMAINS = (
    "1balA00",
    "1bbyA00",
    "1bhuA00",
    "1ux6A01",
    "4bxpA00",
    "3euhA02",
    "4ydzA00",
    "1y1uA01",
    "1zvuA03",
    "1vwxQ00",
    "3eo5A01",
    "1vw4K00",
    "1ytzT00",
)

DEFAULT_DENOISE_PROBE_DOMAINS = (
    "1ux6A01",
    "1vw4K00",
    "1vwxQ00",
    "1balA00",
    "1bbyA00",
    "1bhuA00",
)

DEFAULT_DENOISE_PROBE_DISORDED_DOMAINS = DEFAULT_DENOISE_PROBE_DOMAINS[:3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="biohub/ESMFold2")
    parser.add_argument("--esmc-model", type=Path, default=None)
    parser.add_argument("--esmc-precision", choices=["auto", "bf16", "fp32", "fp8"], default="auto")
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
    #Diagnostics: Pass this flag to train against x_t instead of x_t+dt; leave it off to change back to normal future-frame training.
    parser.add_argument("--target-current-frame", action="store_true", help="diagnostic: use x_t as target_atom_coords instead of the future frame")
    parser.add_argument("--freeze-zero-dt-encoder", action="store_true", help="diagnostic: zero md_conditioning.dt_encoder parameters and freeze them so dt contributes no pair bias")
    parser.add_argument("--num-sampling-steps", type=int, default=1)
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=0, help="save every N optimizer steps; 0 disables step checkpoints")
    parser.add_argument("--save-every-epochs", type=int, default=10, help="save every N epochs; 0 disables epoch checkpoints")
    parser.add_argument("--save-full-checkpoint", action="store_true", help="also save the full model state; default saves md_conditioning only")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="resume md_conditioning and optimizer state from a training checkpoint")
    parser.add_argument("--val-every", type=int, default=100, help="run fixed validation every N optimizer steps; 0 disables validation")
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-length-max", type=int, default=None)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--force-val-domains", default=",".join(DEFAULT_FORCE_VAL_DOMAINS), help="comma-separated domains whose clusters are always assigned to validation")
    parser.add_argument("--force-val-domains-file", type=Path, default=None, help="optional newline-delimited domains whose clusters are always assigned to validation")
    parser.add_argument("--denoise-probe-every", type=int, default=0, help="run fixed x_t -> x_t denoising probe every N optimizer steps; 0 disables")
    parser.add_argument("--denoise-probe-domains", type=int, default=10, help="fallback number of sorted validation domains in the fixed denoising probe when --denoise-probe-domain-list is empty")
    parser.add_argument("--denoise-probe-domain-list", default=",".join(DEFAULT_DENOISE_PROBE_DOMAINS), help="comma-separated validation domains to use for the fixed denoising probe; empty falls back to --denoise-probe-domains")
    parser.add_argument("--denoise-probe-run", default="1", help="HDF5 run/replica key for the fixed denoising probe")
    parser.add_argument("--denoise-probe-frame", type=int, default=0, help="frame index for the fixed denoising probe")
    parser.add_argument("--denoise-probe-delta-frames", type=int, default=1, help="sets dt for the fixed denoising probe")
    parser.add_argument("--denoise-probe-sigmas", default="2,8,16,inference_start", help="comma-separated fixed denoising noise levels in Angstrom; use inference_start for the canonical first sampling sigma")
    parser.add_argument("--denoise-probe-full-inference", action="store_true", help="also run regular diffusion sampling from noise for each probe domain")
    parser.add_argument("--denoise-probe-full-inference-steps", type=int, default=68, help="diffusion sampling steps for --denoise-probe-full-inference")
    parser.add_argument("--denoise-probe-seed", type=int, default=24680, help="base RNG seed for deterministic probe noise/augmentation")
    parser.add_argument("--denoise-probe-dir", default="denoise_probe", help="subdirectory under the run directory for probe outputs")
    parser.add_argument("--denoise-probe-write-pdbs", action="store_true", help="write target/noisy/prediction PDB snapshots for each probe step")
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


def one_record_per_cluster(records: list[dict], rng=random) -> list[dict]:
    by_cluster: dict[str, list[dict]] = {}
    for record in records:
        by_cluster.setdefault(record["cluster"], []).append(record)
    return [rng.choice(group) for group in by_cluster.values()]


def make_smart_batches(records: list[dict], length_max: int, batch_exp: float, rng=random) -> list[list[dict]]:
    sampled = one_record_per_cluster(records, rng=rng)
    sampled = [r for r in sampled if length_to_batch(r["length"], length_max, batch_exp) > 0]
    sampled.sort(key=lambda r: r["length"] + 2.0 * rng.gauss(0.0, 1.0))

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

    rng.shuffle(batches)
    return batches


def parse_domain_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        item.strip()
        for item in value.replace("\n", ",").split(",")
        if item.strip()
    ]


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def forced_validation_domains(args: argparse.Namespace) -> list[str]:
    domains = parse_domain_list(getattr(args, "force_val_domains", ""))
    path = getattr(args, "force_val_domains_file", None)
    if path is not None:
        domains.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return sorted(set(domains))


def denoise_probe_domain_dir_name(domain: str) -> str:
    if domain in DEFAULT_DENOISE_PROBE_DISORDED_DOMAINS:
        return f"{domain}_disorded"
    return domain


def split_train_validation_records(records: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    validation_enabled = args.val_every > 0 and args.val_batches > 0
    probe_enabled = (
        getattr(args, "denoise_probe_every", 0) > 0
        and getattr(args, "denoise_probe_domains", 0) > 0
    )
    if (not validation_enabled and not probe_enabled) or args.val_fraction <= 0:
        return records, []

    by_cluster: dict[str, list[dict]] = {}
    for record in records:
        by_cluster.setdefault(record["cluster"], []).append(record)

    clusters = sorted(by_cluster)
    if len(clusters) < 2:
        return records, []

    rng = random.Random(args.val_seed)
    rng.shuffle(clusters)
    n_val = max(1, int(round(len(clusters) * args.val_fraction)))
    n_val = min(n_val, len(clusters) - 1)
    val_clusters = set(clusters[:n_val])

    by_domain = {record["domain"]: record for record in records}
    missing_forced = []
    forced_clusters = set()
    for domain in forced_validation_domains(args):
        record = by_domain.get(domain)
        if record is None:
            missing_forced.append(domain)
            continue
        forced_clusters.add(record["cluster"])
    if forced_clusters:
        val_clusters.update(forced_clusters)
    if missing_forced:
        print(
            "warning: forced validation domains not found in usable records: "
            + ",".join(missing_forced)
        )

    train_records = [record for record in records if record["cluster"] not in val_clusters]
    val_records = [record for record in records if record["cluster"] in val_clusters]
    return train_records, val_records


def find_axis(shape: tuple[int, ...], size: int, name: str) -> int:
    axes = [axis for axis, axis_size in enumerate(shape) if axis_size == size]
    if len(axes) != 1:
        raise ValueError(f"could not identify {name} axis in shape {shape}")
    return axes[0]


def truncated_geometric(max_value: int, p: float, rng=random) -> int:
    u = rng.random()
    tail = 1.0 - (1.0 - p) ** max_value
    return max(1, min(max_value, math.ceil(math.log(1.0 - u * tail) / math.log(1.0 - p))))


def choose_delta(n_frames: int, max_delta_frames: int, geometric_p: float, rng=random) -> int:
    max_delta = min(max_delta_frames, n_frames - 1)
    if rng.random() < 0.5:
        return truncated_geometric(max_delta, geometric_p, rng=rng)
    return rng.randint(1, max_delta)


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


def tensorize_record(record: dict, args: argparse.Namespace, feature_cache: dict, device: torch.device, rng=random):
    h5py, _ = need_h5py()
    path = record["path"]
    domain = record["domain"]

    with h5py.File(path, "r") as h5:
        group = h5[domain]
        sequence = record["sequence"]
        temperature = args.temperature if args.temperature in group else rng.choice([k for k in group if k.isdigit()])
        run = rng.choice(sorted(group[temperature].keys(), key=int))
        run_group = group[temperature][run]

        coords = run_group["coords"]
        num_atoms = int(group.attrs["numProteinAtoms"])
        num_frames = int(run_group.attrs.get("numFrames", coords.shape[-1]))
        pdb_text = as_text(group["pdbProteinAtoms"][()])

        delta_frames = choose_delta(num_frames, args.max_delta_frames, args.geometric_p, rng=rng)
        frame = rng.randint(0, num_frames - delta_frames - 1)
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
    #Diagnostics: This is the x_t -> x_t diagnostic target. To change back, omit --target-current-frame or remove this branch.
    if args.target_current_frame:
        x_target = x_t.clone()
    else:
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


def tensorize_batch(records: list[dict], args: argparse.Namespace, feature_cache: dict, device: torch.device, rng=random):
    feature_list = []
    extra_list = []
    infos = []

    for record in records:
        features, extra, info = tensorize_record(record, args, feature_cache, device, rng=rng)
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


DENOISE_PROBE_FIELDS = [
    "epoch",
    "step",
    "domain",
    "temperature",
    "run",
    "frame",
    "dt_seconds",
    "probe_sigma",
    "sequence_length",
    "valid_atoms",
    "loss",
    "noise_sigma_mean",
    "model_denoise_rmsd",
    "model_noisy_rmsd",
    "pred_target_rmsd",
    "noisy_target_rmsd",
    "rmsd_improvement",
    "pred_rg",
    "target_rg",
    "ca_ca_mean",
    "ca_ca_min",
    "ca_ca_max",
    "local_bond_bad",
    "local_bond_total",
    "local_bond_bad_fraction",
]

FULL_INFERENCE_FIELDS = [
    "epoch",
    "step",
    "domain",
    "temperature",
    "run",
    "frame",
    "dt_seconds",
    "num_sampling_steps",
    "sequence_length",
    "valid_atoms",
    "rmsd_unaligned",
    "rmsd_kabsch",
    "pred_rg",
    "target_rg",
    "ca_ca_mean",
    "ca_ca_min",
    "ca_ca_max",
    "local_bond_bad",
    "local_bond_total",
    "local_bond_bad_fraction",
]


def sorted_numeric_strings(values) -> list[str]:
    return sorted(values, key=lambda value: int(value) if str(value).isdigit() else str(value))


INFERENCE_START_SIGMA_ALIASES = {
    "inference_start",
    "sampling_start",
    "sample_start",
    "canonical",
}


def canonical_inference_start_sigma(model, device: torch.device) -> float:
    structure_head = getattr(model, "structure_head", None)
    if structure_head is None or not hasattr(structure_head, "inference_noise_schedule"):
        raise RuntimeError("model does not expose structure_head.inference_noise_schedule")

    schedule = structure_head.inference_noise_schedule(
        getattr(structure_head, "inference_num_steps", None),
        device,
    )
    max_inference_sigma = 256.0
    schedule = schedule[schedule <= max_inference_sigma]
    schedule = torch.cat(
        [schedule.new_tensor([max_inference_sigma]), schedule],
        dim=0,
    )
    return float(schedule[0].detach().cpu())


def parse_denoise_probe_sigmas(value: str, model=None, device: torch.device | None = None) -> list[float]:
    sigmas = []
    for part in str(value).split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower() in INFERENCE_START_SIGMA_ALIASES:
            if model is None or device is None:
                raise ValueError(
                    f"{item} can only be resolved after the model is loaded"
                )
            sigmas.append(canonical_inference_start_sigma(model, device))
        else:
            sigmas.append(float(item))
    if not sigmas:
        raise ValueError("--denoise-probe-sigmas must contain at least one value")
    if any(sigma <= 0 for sigma in sigmas):
        raise ValueError("--denoise-probe-sigmas values must be positive")
    return sigmas


def sigma_dir_name(sigma: float) -> str:
    text = f"{float(sigma):g}".replace("-", "m").replace(".", "p")
    return f"sigma_{text}"


def tensorize_fixed_frame_record(
    record: dict,
    args: argparse.Namespace,
    feature_cache: dict,
    device: torch.device,
) -> tuple[dict, dict, dict]:
    h5py, _ = need_h5py()
    path = record["path"]
    domain = record["domain"]

    with h5py.File(path, "r") as h5:
        group = h5[domain]
        sequence = record["sequence"]
        temperature = args.temperature if args.temperature in group else sorted_numeric_strings(k for k in group if k.isdigit())[0]
        if args.denoise_probe_run not in group[temperature]:
            runs = sorted_numeric_strings(group[temperature].keys())
            raise KeyError(
                f"{domain} temperature {temperature} has no run {args.denoise_probe_run}; "
                f"available runs: {runs}"
            )
        run = args.denoise_probe_run
        run_group = group[temperature][run]

        coords = run_group["coords"]
        num_atoms = int(group.attrs["numProteinAtoms"])
        num_frames = int(run_group.attrs.get("numFrames", coords.shape[-1]))
        frame = int(args.denoise_probe_frame)
        if frame < 0 or frame >= num_frames:
            raise IndexError(
                f"{domain} temperature {temperature} run {run} has {num_frames} frames; "
                f"requested frame {frame}"
            )
        pdb_text = as_text(group["pdbProteinAtoms"][()])
        pair = read_two_frames(coords, num_atoms, num_frames, frame, frame)

    if sequence not in feature_cache:
        feature_cache[sequence] = prepare_protein_features(sequence)

    features = {k: v.to(device) for k, v in feature_cache[sequence].items()}
    atom_count = features["ref_pos"].shape[1]
    feature_to_md_atom = feature_to_md_atom_indices(sequence, pdb_text, atom_count)
    current, valid_atoms = map_md_coords_to_features(pair[0], feature_to_md_atom)
    current = center_masked(current, valid_atoms)
    x_t = pad_atom_coords(current, atom_count, device)

    atom_mask = torch.from_numpy(valid_atoms).to(device=device).unsqueeze(0)
    atom_mask &= features["atom_attention_mask"]

    dt_seconds = int(args.denoise_probe_delta_frames) * args.frame_time_ns * 1e-9
    dt = torch.tensor([[dt_seconds]], dtype=torch.float32, device=device)

    extra = {
        "x_t": x_t,
        "dt": dt,
        "target_atom_coords": x_t.clone(),
        "target_atom_mask": atom_mask,
    }
    info = {
        "domain": domain,
        "temperature": temperature,
        "run": run,
        "frame": frame,
        "dt_seconds": dt_seconds,
        "sequence_length": len(sequence),
    }
    return features, extra, info


def build_denoise_probe_items(
    val_records: list[dict],
    args: argparse.Namespace,
    feature_cache: dict,
    device: torch.device,
) -> list[tuple[dict, dict, dict]]:
    if args.denoise_probe_every <= 0 or args.denoise_probe_domains <= 0:
        return []
    if not val_records:
        raise RuntimeError(
            "denoise probe needs validation records; keep --val-fraction > 0"
        )

    requested_domains = unique_in_order(
        parse_domain_list(getattr(args, "denoise_probe_domain_list", ""))
    )
    records_by_domain = {record["domain"]: record for record in val_records}
    if requested_domains:
        missing_domains = [
            domain
            for domain in requested_domains
            if domain not in records_by_domain
        ]
        if missing_domains:
            raise RuntimeError(
                "denoise probe domains are not in validation records: "
                + ",".join(missing_domains)
            )
        records = [records_by_domain[domain] for domain in requested_domains]
    else:
        records = sorted(val_records, key=lambda record: record["domain"])
        records = records[: args.denoise_probe_domains]

    return [
        tensorize_fixed_frame_record(record, args, feature_cache, device)
        for record in records
    ]


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 4:
        pred = pred[:, 0]
    sq = (pred - target).square().sum(dim=-1)
    return (sq * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def squeeze_sample_coords(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim == 4:
        return coords[:, 0]
    return coords


def masked_rmsd_value(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = squeeze_sample_coords(pred).float()
    target = squeeze_sample_coords(target).float()
    mask_f = mask.float()
    sq = (pred - target).square().sum(dim=-1)
    mse = (sq * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return math.sqrt(max(float(mse.detach().cpu()), 0.0))


def kabsch_align_to_target(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred = squeeze_sample_coords(pred).float()
    target = squeeze_sample_coords(target).float()
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=-2, keepdim=True).clamp_min(1e-8)
    pred_mean = (pred * mask_f).sum(dim=-2, keepdim=True) / denom
    target_mean = (target * mask_f).sum(dim=-2, keepdim=True) / denom
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    cov = torch.einsum("bni,bnj->bij", mask_f * target_centered, pred_centered)
    cov32 = cov.float()
    u, _, vh = torch.linalg.svd(
        cov32,
        driver="gesvd" if cov32.is_cuda else None,
    )
    det = torch.linalg.det(u @ vh)
    ones = torch.ones_like(det)
    rot = u @ torch.diag_embed(torch.stack([ones, ones, det], dim=-1)) @ vh
    rot = rot.to(dtype=pred_centered.dtype)
    return pred_centered @ rot.transpose(-1, -2) + target_mean


def radius_of_gyration_value(coords: torch.Tensor, mask: torch.Tensor) -> float:
    coords = squeeze_sample_coords(coords).float()
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    center = (coords * mask_f).sum(dim=1, keepdim=True) / denom
    sq = ((coords - center).square().sum(dim=-1) * mask.float()).sum(dim=1)
    rg = torch.sqrt(sq / mask.float().sum(dim=1).clamp_min(1.0))
    return float(rg.detach().mean().cpu())


def decode_ref_atom_name(chars) -> str:
    return "".join(
        chr(int(char) + 32) if int(char) != 0 else " "
        for char in chars
    ).strip()


def probe_geometry_metrics(
    coords: torch.Tensor,
    features: dict,
    atom_mask: torch.Tensor,
) -> dict:
    _, np = need_h5py()
    coords_np = squeeze_sample_coords(coords)[0].detach().float().cpu().numpy()
    atom_mask_np = atom_mask[0].detach().cpu().numpy().astype(bool)
    atom_to_token = features["atom_to_token"][0].detach().cpu().numpy()
    ref_chars = features["ref_atom_name_chars"][0].detach().cpu().numpy()

    by_token: dict[int, dict[str, object]] = {}
    for atom_idx, valid in enumerate(atom_mask_np):
        if not valid:
            continue
        token_idx = int(atom_to_token[atom_idx])
        name = decode_ref_atom_name(ref_chars[atom_idx])
        by_token.setdefault(token_idx, {})[name] = coords_np[atom_idx]

    local_checks = [
        ("N", "CA", 1.46),
        ("CA", "C", 1.53),
        ("C", "O", 1.24),
        ("CA", "CB", 1.53),
    ]
    bad_bonds = 0
    total_bonds = 0
    tolerance = 0.35
    for atoms in by_token.values():
        for name_a, name_b, ideal in local_checks:
            if name_a not in atoms or name_b not in atoms:
                continue
            distance = float(np.linalg.norm(atoms[name_a] - atoms[name_b]))
            total_bonds += 1
            if abs(distance - ideal) > tolerance:
                bad_bonds += 1

    ca_coords = [
        atoms["CA"]
        for token_idx, atoms in sorted(by_token.items())
        if "CA" in atoms
    ]
    ca_distances = [
        float(np.linalg.norm(ca_coords[idx + 1] - ca_coords[idx]))
        for idx in range(len(ca_coords) - 1)
    ]

    return {
        "ca_ca_mean": sum(ca_distances) / len(ca_distances) if ca_distances else "",
        "ca_ca_min": min(ca_distances) if ca_distances else "",
        "ca_ca_max": max(ca_distances) if ca_distances else "",
        "local_bond_bad": bad_bonds,
        "local_bond_total": total_bonds,
        "local_bond_bad_fraction": bad_bonds / total_bonds if total_bonds else "",
    }


def probe_output_to_pdb_dict(
    features: dict,
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
) -> dict:
    output = {"sample_atom_coords": squeeze_sample_coords(coords).detach().float().cpu()}
    for key in OUTPUT_TO_PDB_FEATURE_KEYS:
        value = features[key].detach().cpu()
        if key == "atom_attention_mask":
            value = atom_mask.detach().cpu().bool()
        output[key] = value
    output["plddt"] = torch.ones_like(
        output["token_attention_mask"], dtype=torch.float32
    )
    return output


def write_probe_pdb(path: Path, features: dict, coords: torch.Tensor, atom_mask: torch.Tensor) -> None:
    path.write_text(output_to_pdb(probe_output_to_pdb_dict(features, coords, atom_mask)))


def pdb_atom_lines_with_chain(pdb_text: str, chain_id: str, serial_start: int) -> tuple[list[str], int]:
    lines = []
    serial = serial_start
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        padded = line.ljust(22)
        lines.append(f"{padded[:6]}{serial:5d}{padded[11:21]}{chain_id}{padded[22:]}")
        serial += 1
    if lines:
        lines.append(f"TER   {serial:5d}      {chain_id}")
        serial += 1
    return lines, serial


def write_probe_overlay_pdb(
    path: Path,
    features: dict,
    target: torch.Tensor,
    pred: torch.Tensor,
    atom_mask: torch.Tensor,
    noisy: torch.Tensor | None = None,
    target_label: str = "target_augmented",
    pred_label: str = "prediction",
    remark: str = "raw denoise overlay; coordinates are not Kabsch aligned",
) -> None:
    entries = [
        ("A", target_label, target),
        ("B", pred_label, pred),
    ]
    if noisy is not None:
        entries.append(("C", "noisy_input", noisy))

    lines = [
        f"REMARK {remark}",
        f"REMARK chain A {target_label}",
        f"REMARK chain B {pred_label}",
    ]
    if noisy is not None:
        lines.append("REMARK chain C noisy_input")

    serial = 1
    for chain_id, _, coords in entries:
        pdb_text = output_to_pdb(probe_output_to_pdb_dict(features, coords, atom_mask))
        atom_lines, serial = pdb_atom_lines_with_chain(pdb_text, chain_id, serial)
        lines.extend(atom_lines)
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_probe_sigma_pdbs(
    domain_dir: Path,
    sigma: float,
    features: dict,
    target: torch.Tensor,
    pred: torch.Tensor,
    atom_mask: torch.Tensor,
    noisy: torch.Tensor | None = None,
) -> None:
    sigma_name = sigma_dir_name(float(sigma))
    write_probe_overlay_pdb(
        domain_dir / f"{sigma_name}_overlay.pdb",
        features,
        target,
        pred,
        atom_mask,
    )
    if noisy is not None:
        write_probe_pdb(
            domain_dir / f"{sigma_name}_noisy.pdb",
            features,
            noisy,
            atom_mask,
        )


def write_probe_reference_pdbs(
    probe_items: list[tuple[dict, dict, dict]],
    probe_dir: Path,
) -> None:
    reference_dir = probe_dir / "reference_pdbs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for features, extra, info in probe_items:
        domain_dir = reference_dir / denoise_probe_domain_dir_name(info["domain"])
        domain_dir.mkdir(parents=True, exist_ok=True)
        write_probe_pdb(
            domain_dir / f"{info['domain']}_run{info['run']}_frame{info['frame']}_target.pdb",
            features,
            extra["target_atom_coords"],
            extra["target_atom_mask"],
        )


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

    raise RuntimeError(
        "model.forward_train(...) did not return a diffusion training loss"
    )


def build_validation_batches(
    val_records: list[dict],
    args: argparse.Namespace,
    feature_cache: dict,
    device: torch.device,
) -> list[tuple[dict, dict, dict]]:
    if not val_records:
        return []

    rng = random.Random(args.val_seed)
    val_length_max = args.val_length_max or args.length_max
    record_batches = make_smart_batches(
        val_records,
        val_length_max,
        args.batch_exp,
        rng=rng,
    )
    record_batches = record_batches[: args.val_batches]
    return [
        tensorize_batch(batch, args, feature_cache, device, rng=rng)
        for batch in record_batches
    ]


def stage1_train_mode(model) -> None:
    model.eval()
    model.md_conditioning.train()


def fork_validation_rng(device: torch.device):
    devices = []
    if device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        devices = [device_index]
    return torch.random.fork_rng(devices=devices)


@torch.no_grad()
def run_validation(
    model,
    validation_batches: list[tuple[dict, dict, dict]],
    args: argparse.Namespace,
    device: torch.device,
    csv_path: Path,
    epoch: int,
    step: int,
) -> dict:
    model.eval()

    loss_sum = 0.0
    atom_weight_sum = 0.0
    denoise_mse_sum = 0.0
    denoise_mse_weight_sum = 0.0
    noisy_mse_sum = 0.0
    noisy_mse_weight_sum = 0.0
    sigma_sum = 0.0
    sigma_weight_sum = 0.0
    sample_count = 0

    for batch_idx, (features, extra, _) in enumerate(validation_batches):
        with fork_validation_rng(device):
            torch.manual_seed(args.val_seed + batch_idx)
            loss, out = training_loss(model, features, extra, args)

        atom_weight = float(extra["target_atom_mask"].float().sum().detach().cpu())
        batch_size = int(extra["target_atom_mask"].shape[0])
        sample_count += batch_size

        loss_sum += float(loss.detach().cpu()) * atom_weight
        atom_weight_sum += atom_weight

        denoise_mse = scalar_if_present(out, "denoise_mse")
        if denoise_mse is not None:
            denoise_mse_sum += denoise_mse * atom_weight
            denoise_mse_weight_sum += atom_weight

        noisy_mse = scalar_if_present(out, "noisy_mse")
        if noisy_mse is not None:
            noisy_mse_sum += noisy_mse * atom_weight
            noisy_mse_weight_sum += atom_weight

        sigma = scalar_if_present(out, "noise_sigma_mean")
        if sigma is not None:
            sigma_sum += sigma * batch_size
            sigma_weight_sum += batch_size

    row = {
        "epoch": epoch,
        "step": step,
        "loss": loss_sum / max(atom_weight_sum, 1.0),
        "denoise_rmsd": "",
        "noisy_rmsd": "",
        "noise_sigma_mean": "",
        "val_batches": len(validation_batches),
        "val_samples": sample_count,
        "val_atoms": atom_weight_sum,
    }
    if denoise_mse_weight_sum:
        row["denoise_rmsd"] = math.sqrt(max(denoise_mse_sum / denoise_mse_weight_sum, 0.0))
    if noisy_mse_weight_sum:
        row["noisy_rmsd"] = math.sqrt(max(noisy_mse_sum / noisy_mse_weight_sum, 0.0))
    if sigma_weight_sum:
        row["noise_sigma_mean"] = sigma_sum / sigma_weight_sum

    append_loss_row(csv_path, row)
    stage1_train_mode(model)

    rmsd_text = (
        f" denoise_rmsd={row['denoise_rmsd']:.5f}"
        if isinstance(row["denoise_rmsd"], float)
        else ""
    )
    print(f"validation epoch={epoch} step={step} loss={row['loss']:.5f}{rmsd_text}")
    return row


def scalar_if_present(out: dict, key: str) -> float | None:
    value = out.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value)


def append_probe_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DENOISE_PROBE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in DENOISE_PROBE_FIELDS})


def append_full_inference_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FULL_INFERENCE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FULL_INFERENCE_FIELDS})


def mean_numeric(rows: list[dict], key: str) -> float | str:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) != "" and row.get(key) is not None
    ]
    if not values:
        return ""
    return sum(values) / len(values)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def run_full_inference_probe_item(
    model,
    features: dict,
    extra: dict,
    info: dict,
    args: argparse.Namespace,
    device: torch.device,
    probe_dir: Path,
    step_dir: Path,
    epoch: int,
    step: int,
    item_idx: int,
) -> dict:
    if not args.denoise_probe_full_inference:
        return {}

    inference_extra = {
        "x_t": extra["x_t"],
        "dt": extra["dt"],
    }
    with fork_validation_rng(device):
        torch.manual_seed(args.denoise_probe_seed + 200_000 + item_idx)
        out = model.forward_train(
            **features,
            **inference_extra,
            num_loops=args.num_loops,
            num_sampling_steps=args.denoise_probe_full_inference_steps,
        )

    pred = out.get("sample_atom_coords")
    if pred is None:
        raise RuntimeError("full inference probe expected sample_atom_coords")
    target = extra["target_atom_coords"]
    mask = extra["target_atom_mask"].bool()
    pred_aligned = kabsch_align_to_target(pred, target, mask)

    row = {
        "epoch": epoch,
        "step": step,
        "domain": info["domain"],
        "temperature": info["temperature"],
        "run": info["run"],
        "frame": info["frame"],
        "dt_seconds": info["dt_seconds"],
        "num_sampling_steps": args.denoise_probe_full_inference_steps,
        "sequence_length": info["sequence_length"],
        "valid_atoms": int(mask.float().sum().detach().cpu()),
        "rmsd_unaligned": masked_rmsd_value(pred, target, mask),
        "rmsd_kabsch": masked_rmsd_value(pred_aligned, target, mask),
        "pred_rg": radius_of_gyration_value(pred_aligned, mask),
        "target_rg": radius_of_gyration_value(target, mask),
        **probe_geometry_metrics(pred_aligned, features, mask),
    }
    append_full_inference_row(probe_dir / "full_inference_metrics.csv", row)

    if args.denoise_probe_write_pdbs:
        domain_dir = step_dir / denoise_probe_domain_dir_name(info["domain"])
        domain_dir.mkdir(parents=True, exist_ok=True)
        write_probe_overlay_pdb(
            domain_dir / "full_inference_overlay.pdb",
            features,
            target,
            pred_aligned,
            mask,
            target_label="run_frame0_target",
            pred_label="full_inference_prediction_kabsch",
            remark=(
                "full inference overlay; chain B is Kabsch aligned to chain A"
            ),
        )

    return row


@torch.no_grad()
def run_denoise_probe(
    model,
    probe_items: list[tuple[dict, dict, dict]],
    args: argparse.Namespace,
    device: torch.device,
    probe_dir: Path,
    epoch: int,
    step: int,
) -> list[dict]:
    if not probe_items:
        return []

    model.eval()
    step_dir = probe_dir / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    full_inference_rows: list[dict] = []
    probe_sigmas = parse_denoise_probe_sigmas(args.denoise_probe_sigmas, model, device)

    for item_idx, (features, extra, info) in enumerate(probe_items):
        with fork_validation_rng(device):
            torch.manual_seed(args.denoise_probe_seed + 100_000 + item_idx)
            fixed_noise = torch.randn_like(extra["target_atom_coords"])

        full_row = run_full_inference_probe_item(
            model,
            features,
            extra,
            info,
            args,
            device,
            probe_dir,
            step_dir,
            epoch,
            step,
            item_idx,
        )
        if full_row:
            full_inference_rows.append(full_row)

        for sigma in probe_sigmas:
            extra_probe = dict(extra)
            extra_probe["denoise_sigma"] = torch.full(
                (extra["target_atom_coords"].shape[0],),
                float(sigma),
                dtype=torch.float32,
                device=device,
            )
            extra_probe["denoise_noise"] = fixed_noise

            with fork_validation_rng(device):
                # Same augmentation for every sigma of this domain/step.
                torch.manual_seed(args.denoise_probe_seed + item_idx)
                loss, out = training_loss(model, features, extra_probe, args)

            pred = out.get("x_pred", out.get("sample_atom_coords"))
            if pred is None:
                raise RuntimeError("denoise probe expected x_pred or sample_atom_coords in model output")
            target = out.get("target_atom_coords_augmented", extra["target_atom_coords"])
            mask = out.get("target_atom_mask", extra["target_atom_mask"]).bool()
            noisy = out.get("x_noisy")

            pred_target_rmsd = masked_rmsd_value(pred, target, mask)
            noisy_target_rmsd = (
                masked_rmsd_value(noisy, target, mask)
                if noisy is not None
                else ""
            )
            row = {
                "epoch": epoch,
                "step": step,
                "domain": info["domain"],
                "temperature": info["temperature"],
                "run": info["run"],
                "frame": info["frame"],
                "dt_seconds": info["dt_seconds"],
                "probe_sigma": float(sigma),
                "sequence_length": info["sequence_length"],
                "valid_atoms": int(mask.float().sum().detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "noise_sigma_mean": scalar_if_present(out, "noise_sigma_mean"),
                "model_denoise_rmsd": scalar_if_present(out, "denoise_rmsd"),
                "model_noisy_rmsd": scalar_if_present(out, "noisy_rmsd"),
                "pred_target_rmsd": pred_target_rmsd,
                "noisy_target_rmsd": noisy_target_rmsd,
                "rmsd_improvement": (
                    noisy_target_rmsd - pred_target_rmsd
                    if isinstance(noisy_target_rmsd, float)
                    else ""
                ),
                "pred_rg": radius_of_gyration_value(pred, mask),
                "target_rg": radius_of_gyration_value(target, mask),
                **probe_geometry_metrics(pred, features, mask),
            }
            rows.append(row)
            append_probe_row(probe_dir / "metrics.csv", row)

            if args.denoise_probe_write_pdbs:
                domain_dir = step_dir / denoise_probe_domain_dir_name(info["domain"])
                domain_dir.mkdir(parents=True, exist_ok=True)
                write_probe_sigma_pdbs(
                    domain_dir,
                    float(sigma),
                    features,
                    target,
                    pred,
                    mask,
                    noisy,
                )

    summary = {
        "epoch": epoch,
        "step": step,
        "domains": sorted({row["domain"] for row in rows}),
        "probe_sigmas": probe_sigmas,
        "mean_pred_target_rmsd": mean_numeric(rows, "pred_target_rmsd"),
        "mean_noisy_target_rmsd": mean_numeric(rows, "noisy_target_rmsd"),
        "mean_rmsd_improvement": mean_numeric(rows, "rmsd_improvement"),
        "mean_noise_sigma": mean_numeric(rows, "noise_sigma_mean"),
        "per_sigma": {
            str(float(sigma)): {
                "mean_pred_target_rmsd": mean_numeric(
                    [row for row in rows if float(row["probe_sigma"]) == float(sigma)],
                    "pred_target_rmsd",
                ),
                "mean_noisy_target_rmsd": mean_numeric(
                    [row for row in rows if float(row["probe_sigma"]) == float(sigma)],
                    "noisy_target_rmsd",
                ),
                "mean_rmsd_improvement": mean_numeric(
                    [row for row in rows if float(row["probe_sigma"]) == float(sigma)],
                    "rmsd_improvement",
                ),
                "local_bond_bad": sum(
                    int(row["local_bond_bad"])
                    for row in rows
                    if float(row["probe_sigma"]) == float(sigma)
                ),
                "local_bond_total": sum(
                    int(row["local_bond_total"])
                    for row in rows
                    if float(row["probe_sigma"]) == float(sigma)
                ),
            }
            for sigma in probe_sigmas
        },
        "local_bond_bad": sum(int(row["local_bond_bad"]) for row in rows),
        "local_bond_total": sum(int(row["local_bond_total"]) for row in rows),
    }
    if full_inference_rows:
        summary["full_inference"] = {
            "num_sampling_steps": args.denoise_probe_full_inference_steps,
            "mean_rmsd_unaligned": mean_numeric(
                full_inference_rows, "rmsd_unaligned"
            ),
            "mean_rmsd_kabsch": mean_numeric(
                full_inference_rows, "rmsd_kabsch"
            ),
            "local_bond_bad": sum(
                int(row["local_bond_bad"]) for row in full_inference_rows
            ),
            "local_bond_total": sum(
                int(row["local_bond_total"]) for row in full_inference_rows
            ),
        }
        if summary["full_inference"]["local_bond_total"]:
            summary["full_inference"]["local_bond_bad_fraction"] = (
                summary["full_inference"]["local_bond_bad"]
                / summary["full_inference"]["local_bond_total"]
            )
    if summary["local_bond_total"]:
        summary["local_bond_bad_fraction"] = (
            summary["local_bond_bad"] / summary["local_bond_total"]
        )
    for sigma_summary in summary["per_sigma"].values():
        if sigma_summary["local_bond_total"]:
            sigma_summary["local_bond_bad_fraction"] = (
                sigma_summary["local_bond_bad"] / sigma_summary["local_bond_total"]
            )
    write_json(step_dir / "summary.json", summary)
    stage1_train_mode(model)

    print(
        "denoise_probe "
        f"epoch={epoch} step={step} "
        f"pred_rmsd={summary['mean_pred_target_rmsd']:.5f} "
        f"bad_bonds={summary['local_bond_bad']}/{summary['local_bond_total']}"
    )
    return rows


def state_dict_to_cpu(state_dict: dict) -> dict:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def tree_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    return value


def serializable_args(args: argparse.Namespace) -> dict:
    result = json.loads(json.dumps(vars(args), default=str))
    #Diagnostics: This metadata makes x_t -> x_t runs explicit in args.json/checkpoints; remove this block when removing --target-current-frame.
    if result.get("target_current_frame"):
        result["diagnostic_target"] = "target_atom_coords = x_t (current input frame), not x_t+dt"
    return result


def save_checkpoint(path: Path, model, optimizer, args: argparse.Namespace, epoch: int, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "checkpoint_kind": "md_conditioning",
        "md_conditioning": state_dict_to_cpu(model.md_conditioning.state_dict()),
        "trainable_parameter_names": [
            name for name, param in model.named_parameters() if param.requires_grad
        ],
        "optimizer": tree_to_cpu(optimizer.state_dict()),
        "args": serializable_args(args),
    }
    if args.save_full_checkpoint:
        checkpoint["checkpoint_kind"] = "full_model"
        checkpoint["model"] = state_dict_to_cpu(model.state_dict())
    torch.save(checkpoint, path)


def load_training_checkpoint(path: Path, model, optimizer, device: torch.device) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    if "md_conditioning" not in checkpoint:
        raise RuntimeError(f"{path} does not contain md_conditioning weights")
    model.md_conditioning.load_state_dict(checkpoint["md_conditioning"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))


def run_dir_from_resume_checkpoint(path: Path) -> Path | None:
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return None


def append_loss_row(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def resolve_esmc_precision(device: torch.device, requested: str) -> str:
    if requested != "auto":
        return requested
    return "bf16" if device.type == "cuda" else "fp32"


def load_model(args: argparse.Namespace, device: torch.device):
    esmc_precision = resolve_esmc_precision(device, args.esmc_precision)
    if args.esmc_model is None:
        model = ESMFold2Model.from_pretrained(args.model, esmc_precision=esmc_precision)
        return model.to(device)

    model = ESMFold2Model.from_pretrained(args.model, load_esmc=False)
    model.to(device)
    model.load_esmc(str(args.esmc_model), precision=esmc_precision)
    return model


def zero_and_freeze_dt_encoder(model) -> int:
    dt_encoder = getattr(getattr(model, "md_conditioning", None), "dt_encoder", None)
    if dt_encoder is None:
        raise RuntimeError("Expected model.md_conditioning.dt_encoder")

    param_count = 0
    with torch.no_grad():
        for param in dt_encoder.parameters():
            param.zero_()
            param.requires_grad_(False)
            param_count += param.numel()
    return param_count


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)

    records = load_domain_records(args.data_dir, args.temperature)
    train_records, val_records = split_train_validation_records(records, args)
    if not train_records:
        raise RuntimeError("validation split left no training records")

    resume_run_dir = (
        run_dir_from_resume_checkpoint(args.resume_checkpoint)
        if args.resume_checkpoint is not None
        else None
    )
    if resume_run_dir is None:
        run_name = "md_esmfold2_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.out_dir / run_name
    else:
        run_dir = resume_run_dir
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(serializable_args(args), indent=2, default=str) + "\n")
    if val_records:
        (run_dir / "validation_domains.txt").write_text(
            "\n".join(record["domain"] for record in sorted(val_records, key=lambda r: r["domain"])) + "\n"
        )

    model = load_model(args, device)
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
    frozen_dt_params = 0
    if args.freeze_zero_dt_encoder:
        frozen_dt_params = zero_and_freeze_dt_encoder(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after enabling md_conditioning")
    trainable_count = sum(param.numel() for param in trainable_params)
    total_count = sum(param.numel() for param in model.parameters())
    stage1_train_mode(model)
    print(
        f"trainable_params={trainable_count} total_params={total_count} "
        "trainable_module=md_conditioning"
    )
    if args.freeze_zero_dt_encoder:
        print(
            "dt_encoder_zero_frozen=true "
            f"frozen_dt_params={frozen_dt_params}"
        )
    print(
        f"records train={len(train_records)} validation={len(val_records)} "
        f"validation_every={args.val_every}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint is not None:
        start_epoch, global_step = load_training_checkpoint(
            args.resume_checkpoint, model, optimizer, device
        )
        if args.freeze_zero_dt_encoder:
            zero_and_freeze_dt_encoder(model)
        print(
            f"resumed_checkpoint={args.resume_checkpoint} "
            f"start_epoch={start_epoch} global_step={global_step}"
        )
    feature_cache: dict = {}
    loss_csv = run_dir / "loss.csv"
    validation_csv = run_dir / "validation.csv"
    validation_batches = build_validation_batches(val_records, args, feature_cache, device)
    if val_records and not validation_batches:
        print("validation disabled: no validation batches fit val_length_max")

    denoise_probe_dir = run_dir / args.denoise_probe_dir
    denoise_probe_items = build_denoise_probe_items(
        val_records,
        args,
        feature_cache,
        device,
    )
    if denoise_probe_items:
        denoise_probe_dir.mkdir(parents=True, exist_ok=True)
        probe_domains = [info["domain"] for _, _, info in denoise_probe_items]
        (denoise_probe_dir / "probe_domains.txt").write_text(
            "\n".join(probe_domains) + "\n"
        )
        write_probe_reference_pdbs(denoise_probe_items, denoise_probe_dir)
        write_json(
            denoise_probe_dir / "config.json",
            {
                "every": args.denoise_probe_every,
                "domains": probe_domains,
                "run": args.denoise_probe_run,
                "frame": args.denoise_probe_frame,
                "delta_frames": args.denoise_probe_delta_frames,
                "dt_seconds": args.denoise_probe_delta_frames * args.frame_time_ns * 1e-9,
                "sigmas": parse_denoise_probe_sigmas(args.denoise_probe_sigmas, model, device),
                "full_inference": {
                    "enabled": args.denoise_probe_full_inference,
                    "num_sampling_steps": args.denoise_probe_full_inference_steps,
                    "overlay": (
                        "full_inference_overlay.pdb contains chain A run_frame0_target "
                        "and chain B Kabsch-aligned full_inference_prediction"
                    ),
                },
                "seed": args.denoise_probe_seed,
                "write_pdbs": args.denoise_probe_write_pdbs,
                "note": (
                    "Fixed x_t -> x_t probe. The probe passes explicit sigma/noise "
                    "through forward_train and saves noisy/prediction PDBs per sigma."
                ),
            },
        )

        run_denoise_probe(
            model,
            denoise_probe_items,
            args,
            device,
            denoise_probe_dir,
            epoch=start_epoch,
            step=global_step,
        )

    if validation_batches and args.resume_checkpoint is None:
        run_validation(
            model,
            validation_batches,
            args,
            device,
            validation_csv,
            epoch=0,
            step=global_step,
        )

    if start_epoch >= args.epochs:
        raise RuntimeError(
            f"resume checkpoint is already at epoch {start_epoch}, "
            f"but --epochs is {args.epochs}; pass a larger total epoch count"
        )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        batches = make_smart_batches(train_records, args.length_max, args.batch_exp)
        if args.steps_per_epoch > 0:
            batches = batches[: args.steps_per_epoch]

        sample_count = sum(len(batch) for batch in batches)
        print(f"epoch={epoch} batches={len(batches)} samples={sample_count}")

        for batch_records in batches:
            optimizer.zero_grad(set_to_none=True)
            features, extra, info = tensorize_batch(batch_records, args, feature_cache, device)
            loss, out = training_loss(model, features, extra, args)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            global_step += 1

            row = {
                "epoch": epoch,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                **info,
            }
            for key in ("denoise_rmsd", "noisy_rmsd", "noise_sigma_mean"):
                value = scalar_if_present(out, key)
                if value is not None:
                    row[key] = value
            append_loss_row(loss_csv, row)

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"step_{global_step:07d}.pt", model, optimizer, args, epoch, global_step)
                print(f"epoch={epoch} step={global_step} loss={row['loss']:.5f}")

            if validation_batches and args.val_every > 0 and global_step % args.val_every == 0:
                run_validation(
                    model,
                    validation_batches,
                    args,
                    device,
                    validation_csv,
                    epoch=epoch,
                    step=global_step,
                )

            if denoise_probe_items and global_step % args.denoise_probe_every == 0:
                run_denoise_probe(
                    model,
                    denoise_probe_items,
                    args,
                    device,
                    denoise_probe_dir,
                    epoch=epoch,
                    step=global_step,
                )

        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", model, optimizer, args, epoch, global_step)
            print(f"saved checkpoint epoch={epoch} step={global_step}")

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, args, args.epochs, global_step)
    print(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
