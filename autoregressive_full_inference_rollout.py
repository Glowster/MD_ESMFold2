#!/usr/bin/env python3
"""Autoregressive full-inference rollout for the MD-conditioned ESMFold2 probe set.

At rollout step 1, the model receives the true run 1 frame 0 structure as x_t.
After that, each 68-step full-inference prediction is fed back as the next x_t.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch

import train_md_esmfold2 as train


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs" / "md_esmfold2_20260701_192013"
DOMAINS = ("1ux6A01", "1vw4K00", "1vwxQ00", "1balA00", "1bbyA00", "1bhuA00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=ROOT / "hf_models" / "biohub_ESMFold2")
    parser.add_argument("--lm-z-cache-dir", type=Path, default=ROOT / "lm_z_cache")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mdcath_320K_len_le200" / "data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-steps", type=int, default=2000)
    parser.add_argument("--diffusion-steps", type=int, default=68)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=24680)
    return parser.parse_args()


def write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_multimodel_pdb(
    path: Path,
    model_number: int,
    features: dict,
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
) -> None:
    if not path.exists():
        path.write_text(
            "REMARK autoregressive full-inference rollout trajectory\n"
            "REMARK one MODEL record per rollout step\n"
            "REMARK coordinates are Kabsch aligned to run 1 frame 0 for viewing\n"
        )
    pdb_text = train.output_to_pdb(
        train.probe_output_to_pdb_dict(features, coords, atom_mask)
    )
    atom_lines, _ = train.pdb_atom_lines_with_chain(pdb_text, "B", 1)
    with path.open("a") as handle:
        handle.write(f"MODEL     {model_number:4d}\n")
        for line in atom_lines:
            handle.write(line + "\n")
        handle.write("ENDMDL\n")


def load_probe_batch(args: argparse.Namespace, device: torch.device):
    helper_args = SimpleNamespace(
        model=args.model,
        esmc_model=None,
        esmc_precision="auto",
        precomputed_lm_z_only=True,
        require_lm_z_cache=True,
        lm_z_cache_dir=args.lm_z_cache_dir,
        lm_z_cache_dtype="bf16",
        _lm_z_cache_index=train.load_lm_z_cache_index(args.lm_z_cache_dir),
        data_dir=args.data_dir,
        temperature="320",
        frame_time_ns=1.0,
        denoise_probe_run="1",
        denoise_probe_frame=0,
        denoise_probe_delta_frames=1,
        denoise_probe_domain_list=",".join(DOMAINS),
        denoise_probe_domains=len(DOMAINS),
        denoise_probe_every=1,
        val_every=100,
        val_batches=4,
        val_fraction=0.05,
        val_seed=12345,
        force_val_domains=",".join(train.DEFAULT_FORCE_VAL_DOMAINS),
        force_val_domains_file=None,
    )

    records = train.load_domain_records(helper_args.data_dir, helper_args.temperature)
    train_records, val_records = train.split_train_validation_records(records, helper_args)
    train.assert_forced_validation_domains_not_in_train(train_records, helper_args)
    records_by_domain = {record["domain"]: record for record in val_records}

    feature_cache = {}
    features_list, extras_list, infos = [], [], []
    for domain in DOMAINS:
        record = records_by_domain[domain]
        features, extra, info = train.tensorize_fixed_frame_record(
            record, helper_args, feature_cache, device
        )
        features["lm_z"] = train.load_cached_lm_z(record, record["sequence"], helper_args, device)
        features_list.append(features)
        extras_list.append(extra)
        infos.append(info)

    return helper_args, train.pad_features(features_list), train.pad_extras(extras_list), features_list, extras_list, infos


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = args.checkpoint or args.run_dir / "checkpoints" / "step_0001000.pt"
    out_dir = args.out_dir or args.run_dir / "autoregressive_full_inference_step1000"
    out_dir.mkdir(parents=True, exist_ok=True)

    helper_args, features, extras, one_features, one_extras, infos = load_probe_batch(args, device)
    model = train.load_model(helper_args, device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.md_conditioning.load_state_dict(checkpoint_data["md_conditioning"])
    model.eval()

    current_x = extras["x_t"]
    dt = extras["dt"]

    for step in range(1, args.rollout_steps + 1):
        with torch.inference_mode(), train.fork_validation_rng(device):
            torch.manual_seed(args.seed + step)
            out = model.forward_train(
                **features,
                x_t=current_x,
                dt=dt,
                num_loops=1,
                num_sampling_steps=args.diffusion_steps,
            )
        pred = train.squeeze_sample_coords(out["sample_atom_coords"]).detach()
        current_x = pred

        for item_idx, info in enumerate(infos):
            atom_count = one_extras[item_idx]["x_t"].shape[1]
            pred_i = pred[item_idx : item_idx + 1, :atom_count]
            target_i = one_extras[item_idx]["target_atom_coords"]
            mask_i = one_extras[item_idx]["target_atom_mask"].bool()
            aligned_i = train.kabsch_align_to_target(pred_i, target_i, mask_i)
            features_i = one_features[item_idx]

            row = {
                "rollout_step": step,
                "domain": info["domain"],
                "diffusion_steps": args.diffusion_steps,
                "rmsd_unaligned": train.masked_rmsd_value(pred_i, target_i, mask_i),
                "rmsd_kabsch": train.masked_rmsd_value(aligned_i, target_i, mask_i),
                "pred_rg": train.radius_of_gyration_value(aligned_i, mask_i),
                "target_rg": train.radius_of_gyration_value(target_i, mask_i),
                **train.probe_geometry_metrics(aligned_i, features_i, mask_i),
            }
            write_csv(out_dir / "metrics.csv", row)

            domain_dir = out_dir / f"{info['domain']}"
            domain_dir.mkdir(parents=True, exist_ok=True)
            if step == 1:
                train.write_probe_pdb(
                    domain_dir / "run1_frame0_target.pdb",
                    features_i,
                    target_i,
                    mask_i,
                )
            append_multimodel_pdb(
                domain_dir / "rollout_predictions_kabsch_multimodel.pdb",
                step,
                features_i,
                aligned_i,
                mask_i,
            )

            if step == 1 or step % args.write_every == 0 or step == args.rollout_steps:
                train.write_probe_overlay_pdb(
                    domain_dir / f"rollout_step_{step:04d}_overlay.pdb",
                    features_i,
                    target_i,
                    aligned_i,
                    mask_i,
                    target_label="run1_frame0_target",
                    pred_label="autoregressive_full_inference_kabsch",
                    remark="autoregressive rollout overlay; chain B is Kabsch aligned for viewing only",
                )

        if step % 10 == 0:
            print(f"rollout_step={step}/{args.rollout_steps}", flush=True)

    print(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
