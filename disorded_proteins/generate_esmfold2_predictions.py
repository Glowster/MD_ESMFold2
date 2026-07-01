#!/usr/bin/env python3
"""Generate full ESMFold2 predictions for the ranked disordered mdCATH domains.

This uses the local full ESMFold2 weights and explicitly loads the local ESMC-6B
weights so it does not need Hugging Face network access.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from ctypes import (
    CDLL,
    POINTER,
    byref,
    c_char_p,
    c_int,
    c_longlong,
    c_size_t,
    c_uint,
    c_void_p,
    create_string_buffer,
)
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "mdcath_320K_len_le200" / "data"
DEFAULT_MODEL_DIR = ROOT / "hf_models" / "biohub_ESMFold2"
DEFAULT_ESMC_DIR = ROOT / "hf_models" / "biohub_ESMC-6B"
DEFAULT_TRANSFORMERS_SRC = ROOT / "transformers" / "src"


class H5Strings:
    def __init__(self) -> None:
        self.lib = CDLL("/lib/x86_64-linux-gnu/libhdf5_serial.so.103")
        hid_t = c_longlong
        specs = [
            ("H5open", c_int, []),
            ("H5Fopen", hid_t, [c_char_p, c_uint, hid_t]),
            ("H5Fclose", c_int, [hid_t]),
            ("H5Dopen2", hid_t, [hid_t, c_char_p, hid_t]),
            ("H5Dclose", c_int, [hid_t]),
            ("H5Dget_type", hid_t, [hid_t]),
            ("H5Dread", c_int, [hid_t, hid_t, hid_t, hid_t, hid_t, c_void_p]),
            ("H5Tget_size", c_size_t, [hid_t]),
            ("H5Tis_variable_str", c_int, [hid_t]),
            ("H5Tclose", c_int, [hid_t]),
            ("H5free_memory", c_int, [c_void_p]),
            ("H5Eset_auto2", c_int, [hid_t, c_void_p, c_void_p]),
        ]
        for name, restype, argtypes in specs:
            fn = getattr(self.lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
        self.lib.H5open()
        self.lib.H5Eset_auto2(0, None, None)

    def read_string(self, h5_path: Path, dataset_path: str) -> str:
        file_id = self.lib.H5Fopen(str(h5_path).encode(), 0, 0)
        if file_id < 0:
            raise RuntimeError(f"could not open {h5_path}")
        dset = self.lib.H5Dopen2(file_id, dataset_path.encode(), 0)
        if dset < 0:
            self.lib.H5Fclose(file_id)
            raise RuntimeError(f"could not open {dataset_path} in {h5_path}")
        dtype = self.lib.H5Dget_type(dset)
        try:
            if self.lib.H5Tis_variable_str(dtype) > 0:
                ptr = c_char_p()
                status = self.lib.H5Dread(dset, dtype, 0, 0, 0, byref(ptr))
                data = ptr.value or b""
                addr = c_void_p.from_buffer(ptr).value
                if addr:
                    self.lib.H5free_memory(c_void_p(addr))
            else:
                size = self.lib.H5Tget_size(dtype)
                buf = create_string_buffer(size)
                status = self.lib.H5Dread(dset, dtype, 0, 0, 0, buf)
                data = buf.raw.rstrip(b"\x00")
            if status < 0:
                raise RuntimeError(f"could not read {dataset_path}")
            return data.decode("utf-8", "replace").strip()
        finally:
            self.lib.H5Tclose(dtype)
            self.lib.H5Dclose(dset)
            self.lib.H5Fclose(file_id)


def read_domains(path: Path) -> list[str]:
    if path.suffix == ".tsv":
        with path.open() as handle:
            return [row["domain"] for row in csv.DictReader(handle, delimiter="\t")]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=Path, default=Path(__file__).with_name("domains.txt"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "esmfold2_predictions")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--esmc-dir", type=Path, default=DEFAULT_ESMC_DIR)
    parser.add_argument("--transformers-src", type=Path, default=DEFAULT_TRANSFORMERS_SRC)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--esmc-precision", choices=["bf16", "fp32", "fp8"], default="bf16")
    parser.add_argument("--predictions-per-domain", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.transformers_src))

    try:
        import torch
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    except Exception as err:
        raise RuntimeError(
            "ESMFold2 prediction requires torch/safetensors and the local transformers source environment"
        ) from err

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    domains = read_domains(args.domains)
    h5 = H5Strings()
    sequences = {
        domain: h5.read_string(args.data_dir / f"mdcath_dataset_{domain}.h5", f"{domain}/sequence")
        for domain in domains
    }

    model = ESMFold2Model.from_pretrained(str(args.model_dir), load_esmc=False)
    model = model.to(args.device).eval()
    model.load_esmc(str(args.esmc_dir), precision=args.esmc_precision)

    manifest_path = args.out_dir / "prediction_manifest.tsv"
    with manifest_path.open("w", newline="") as manifest_handle:
        writer = csv.DictWriter(
            manifest_handle,
            delimiter="\t",
            fieldnames=["domain", "prediction_index", "seed", "sequence_length", "pdb_path"],
        )
        writer.writeheader()

        for domain in domains:
            seq = sequences[domain]
            domain_dir = args.out_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            for pred_idx in range(1, args.predictions_per_domain + 1):
                seed = args.seed + (1000 * domains.index(domain)) + pred_idx - 1
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                pdb = model.infer_protein_as_pdb(
                    seq,
                    num_loops=args.num_loops,
                    num_sampling_steps=args.num_sampling_steps,
                    num_diffusion_samples=args.num_diffusion_samples,
                )
                out_path = domain_dir / f"{domain}_esmfold2_full_pred_{pred_idx:02d}_seed_{seed}.pdb"
                out_path.write_text(pdb)
                writer.writerow(
                    {
                        "domain": domain,
                        "prediction_index": pred_idx,
                        "seed": seed,
                        "sequence_length": len(seq),
                        "pdb_path": out_path.relative_to(args.out_dir.parent),
                    }
                )
                manifest_handle.flush()
                print(f"wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
