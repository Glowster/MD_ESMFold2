#!/usr/bin/env python3
"""Precompute deterministic ESM-C-derived lm_z tensors for MD ESMFold2 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
LOCAL_TRANSFORMERS = ROOT / "transformers" / "src"
sys.path.insert(0, str(LOCAL_TRANSFORMERS))

from transformers.models.esmfold2.protein_utils import prepare_protein_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "hf_models" / "biohub_ESMFold2")
    parser.add_argument("--esmc-model", type=Path, default=ROOT / "hf_models" / "biohub_ESMC-6B")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mdcath_320K_len_le200" / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "lm_z_cache")
    parser.add_argument("--temperature", default="320")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--esmc-precision", choices=["auto", "bf16", "fp32", "fp8"], default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of unique sequences to precompute; 0 means all")
    parser.add_argument("--domain-list", default="", help="comma-separated domains to precompute; empty means all")
    parser.add_argument("--domain-list-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    return parser.parse_args()


def resolve_esmc_precision(device: torch.device, requested: str) -> str:
    if requested != "auto":
        return requested
    return "bf16" if device.type == "cuda" else "fp32"


def target_dtype(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return as_text(value.tolist())
    if isinstance(value, list):
        return "".join(as_text(v) for v in value)
    return str(value)


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def domain_from_path(path: Path) -> str:
    name = path.stem
    prefix = "mdcath_dataset_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def load_domain_records(data_dir: Path, temperature: str) -> list[dict]:
    try:
        import h5py
    except ImportError as err:
        raise RuntimeError("Install h5py before precomputing lm_z") from err

    records = []
    for path in sorted(data_dir.glob("mdcath_dataset_*.h5")):
        domain = domain_from_path(path)
        with h5py.File(path, "r") as h5:
            group = h5[domain]
            sequence = as_text(group["sequence"][()])
            available_temperatures = [key for key in group.keys() if key.isdigit()]
            if temperature not in available_temperatures and not available_temperatures:
                continue
            records.append(
                {
                    "path": path,
                    "domain": domain,
                    "cluster": as_text(group.attrs.get("cluster", domain)),
                    "length": len(sequence),
                    "sequence": sequence,
                }
            )
    if not records:
        raise FileNotFoundError(f"no mdCATH HDF5 files found under {data_dir}")
    return records


def parse_domain_list(value: str, path: Path | None) -> set[str]:
    domains: list[str] = []
    if value:
        domains.extend(item.strip() for item in value.split(",") if item.strip())
    if path is not None:
        domains.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return set(domains)


def configure_determinism(seed: int) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "use_deterministic_algorithms": True,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(path: Path) -> str | None:
    return file_sha256(path) if path.exists() else None


def directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = item.stat()
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(cpu_tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def save_lm_z(path: Path, lm_z: torch.Tensor, metadata: dict[str, str]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as err:
        raise RuntimeError("Install safetensors before precomputing lm_z") from err

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_file({"lm_z": lm_z.contiguous().cpu()}, str(tmp_path), metadata=metadata)
    os.replace(tmp_path, path)


def load_lm_z(path: Path) -> torch.Tensor:
    try:
        from safetensors.torch import load_file
    except ImportError as err:
        raise RuntimeError("Install safetensors before verifying lm_z") from err
    return load_file(str(path), device="cpu")["lm_z"]


def tensor_features(sequence: str, device: torch.device) -> dict[str, torch.Tensor]:
    features = prepare_protein_features(sequence)
    keys = ("input_ids", "asym_id", "residue_index", "mol_type", "token_attention_mask")
    return {key: features[key].to(device) for key in keys}


def compute_one_lm_z(
    model,
    sequence: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    features = tensor_features(sequence, device)
    lm_z = model.compute_lm_z(
        input_ids=features["input_ids"],
        asym_id=features["asym_id"],
        residue_index=features["residue_index"],
        mol_type=features["mol_type"],
        token_attention_mask=features["token_attention_mask"],
        lm_mask_pct=0.0,
    )
    return lm_z[0].to(dtype=dtype).cpu()


def unique_sequence_records(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique_records = []
    for record in records:
        sequence = record["sequence"]
        seq_hash = sequence_sha256(sequence)
        if seq_hash in seen:
            continue
        seen.add(seq_hash)
        unique_records.append(record)
    return unique_records


def build_manifest(args: argparse.Namespace, model, esmc_precision: str, deterministic_flags: dict) -> dict:
    model_path = Path(args.model)
    esmc_path = Path(args.esmc_model)
    return {
        "cache_kind": "esmfold2_lm_z",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dtype": args.dtype,
        "model_path": str(model_path),
        "esmc_path": str(esmc_path),
        "esmc_precision": esmc_precision,
        "lm_mask_pct": 0.0,
        "esmfold2_config_hash": optional_file_sha256(model_path / "config.json"),
        "esmfold2_language_model_state_hash": state_dict_sha256(model.language_model.state_dict()),
        "esmc_directory_fingerprint": directory_fingerprint(esmc_path),
        "transformers_source_git_sha": git_output(["-C", "transformers", "rev-parse", "HEAD"]),
        "workspace_git_sha": git_output(["rev-parse", "HEAD"]),
        "workspace_git_dirty": bool(git_output(["status", "--porcelain"])),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": args.device,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() and torch.device(args.device).type == "cuda" else "",
        "deterministic_flags": deterministic_flags,
    }


def verify_rows(rows: list[dict], model, device: torch.device, dtype: torch.dtype, cache_dir: Path, count: int, seed: int) -> None:
    if count <= 0 or not rows:
        return
    rng = random.Random(seed)
    sample = rng.sample(rows, k=min(count, len(rows)))
    for row in sample:
        tensor_path = cache_dir / row["relative_tensor_path"]
        cached = load_lm_z(tensor_path)
        recomputed = compute_one_lm_z(model, row["sequence"], device, dtype)
        if not torch.equal(cached, recomputed):
            raise RuntimeError(
                f"deterministic verification failed for {row['domain']} "
                f"at {tensor_path}"
            )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    deterministic_flags = {}
    if not args.allow_nondeterministic:
        deterministic_flags = configure_determinism(args.seed)

    selected_domains = parse_domain_list(args.domain_list, args.domain_list_file)
    records = load_domain_records(args.data_dir, args.temperature)
    if selected_domains:
        records = [record for record in records if record["domain"] in selected_domains]
        missing = sorted(selected_domains - {record["domain"] for record in records})
        if missing:
            raise RuntimeError("requested domains not found: " + ",".join(missing))
    records = unique_sequence_records(records)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("no records selected for lm_z precompute")

    esmc_precision = resolve_esmc_precision(device, args.esmc_precision)
    dtype = target_dtype(args.dtype)

    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(str(args.model), load_esmc=False)
    model.to(device)
    model.load_esmc(str(args.esmc_model), precision=esmc_precision)
    model.eval()

    cache_dir = args.out_dir
    tensor_dir = cache_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, record in enumerate(records, start=1):
        sequence = record["sequence"]
        seq_hash = sequence_sha256(sequence)
        rel_path = Path("tensors") / f"{record['domain']}.{seq_hash[:16]}.safetensors"
        tensor_path = cache_dir / rel_path

        if tensor_path.exists() and not args.overwrite:
            tensor_hash = file_sha256(tensor_path)
        else:
            lm_z = compute_one_lm_z(model, sequence, device, dtype)
            save_lm_z(
                tensor_path,
                lm_z,
                metadata={
                    "domain": record["domain"],
                    "sequence_sha256": seq_hash,
                    "dtype": args.dtype,
                    "lm_mask_pct": "0.0",
                },
            )
            tensor_hash = file_sha256(tensor_path)

        rows.append(
            {
                "domain": record["domain"],
                "sequence": sequence,
                "sequence_sha256": seq_hash,
                "length": len(sequence),
                "relative_tensor_path": rel_path.as_posix(),
                "tensor_sha256": tensor_hash,
            }
        )
        if idx == 1 or idx % 25 == 0 or idx == len(records):
            print(f"precomputed_lm_z={idx}/{len(records)} domain={record['domain']} length={len(sequence)}")

    manifest = build_manifest(args, model, esmc_precision, deterministic_flags)
    manifest["num_records"] = len(rows)
    manifest["sum_l"] = sum(row["length"] for row in rows)
    manifest["sum_l2"] = sum(row["length"] * row["length"] for row in rows)

    verify_rows(rows, model, device, dtype, cache_dir, args.verify_samples, args.seed)

    atomic_write_text(
        cache_dir / "index.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    atomic_write_text(
        cache_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(f"wrote_lm_z_cache={cache_dir} records={len(rows)} dtype={args.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
