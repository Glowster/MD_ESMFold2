#!/usr/bin/env python3
"""Export mdCATH trajectories as multi-model PDB files without h5py.

Each output PDB contains MODEL/ENDMDL records for every selected frame from
each requested run. Coordinates come from the HDF5 `coords` dataset and atom
metadata comes from `pdbProteinAtoms`.
"""

from __future__ import annotations

import argparse
import csv
from ctypes import (
    CDLL,
    POINTER,
    byref,
    c_char_p,
    c_int,
    c_longlong,
    c_size_t,
    c_uint,
    c_ulonglong,
    c_void_p,
    create_string_buffer,
)
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "mdcath_320K_len_le200" / "data"


class H5:
    def __init__(self) -> None:
        self.lib = CDLL("/lib/x86_64-linux-gnu/libhdf5_serial.so.103")
        hid_t = c_longlong
        hsize_t = c_ulonglong
        specs = [
            ("H5open", c_int, []),
            ("H5Fopen", hid_t, [c_char_p, c_uint, hid_t]),
            ("H5Fclose", c_int, [hid_t]),
            ("H5Gopen2", hid_t, [hid_t, c_char_p, hid_t]),
            ("H5Gclose", c_int, [hid_t]),
            ("H5Lexists", c_int, [hid_t, c_char_p, hid_t]),
            ("H5Dopen2", hid_t, [hid_t, c_char_p, hid_t]),
            ("H5Dclose", c_int, [hid_t]),
            ("H5Dget_space", hid_t, [hid_t]),
            ("H5Dget_type", hid_t, [hid_t]),
            ("H5Dread", c_int, [hid_t, hid_t, hid_t, hid_t, hid_t, c_void_p]),
            ("H5Sget_simple_extent_ndims", c_int, [hid_t]),
            ("H5Sget_simple_extent_dims", c_int, [hid_t, POINTER(hsize_t), POINTER(hsize_t)]),
            ("H5Sselect_hyperslab", c_int, [hid_t, c_int, POINTER(hsize_t), POINTER(hsize_t), POINTER(hsize_t), POINTER(hsize_t)]),
            ("H5Screate_simple", hid_t, [c_int, POINTER(hsize_t), POINTER(hsize_t)]),
            ("H5Sclose", c_int, [hid_t]),
            ("H5Aopen", hid_t, [hid_t, c_char_p, hid_t]),
            ("H5Aread", c_int, [hid_t, hid_t, c_void_p]),
            ("H5Aclose", c_int, [hid_t]),
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
        self.h5t_native_llong = c_longlong.in_dll(self.lib, "H5T_NATIVE_LLONG_g").value
        self.h5t_native_float = c_longlong.in_dll(self.lib, "H5T_NATIVE_FLOAT_g").value

    def exists(self, file_id: int, path: str) -> bool:
        return self.lib.H5Lexists(file_id, path.encode(), 0) > 0

    def read_int_attr(self, obj_id: int, name: str) -> int | None:
        attr = self.lib.H5Aopen(obj_id, name.encode(), 0)
        if attr < 0:
            return None
        value = c_longlong()
        try:
            status = self.lib.H5Aread(attr, self.h5t_native_llong, byref(value))
            if status < 0:
                return None
            return int(value.value)
        finally:
            self.lib.H5Aclose(attr)

    def read_string(self, file_id: int, path: str) -> str:
        dset = self.lib.H5Dopen2(file_id, path.encode(), 0)
        if dset < 0:
            raise RuntimeError(f"could not open dataset {path}")
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
                raise RuntimeError(f"could not read dataset {path}")
            return data.decode("utf-8", "replace")
        finally:
            self.lib.H5Tclose(dtype)
            self.lib.H5Dclose(dset)

    def dataset_shape(self, dset: int) -> tuple[int, ...]:
        space = self.lib.H5Dget_space(dset)
        ndims = self.lib.H5Sget_simple_extent_ndims(space)
        dims = (c_ulonglong * ndims)()
        maxdims = (c_ulonglong * ndims)()
        self.lib.H5Sget_simple_extent_dims(space, dims, maxdims)
        self.lib.H5Sclose(space)
        return tuple(int(dims[i]) for i in range(ndims))

    def make_frame_reader(self, file_id: int, path: str, num_atoms: int, num_frames: int):
        dset = self.lib.H5Dopen2(file_id, path.encode(), 0)
        if dset < 0:
            raise RuntimeError(f"could not open coords dataset {path}")
        shape = self.dataset_shape(dset)
        frame_axes = [axis for axis, size in enumerate(shape) if size == num_frames]
        atom_axes = [axis for axis, size in enumerate(shape) if size == num_atoms]
        xyz_axes = [axis for axis, size in enumerate(shape) if size == 3]
        if len(frame_axes) != 1 or len(atom_axes) != 1 or len(xyz_axes) != 1:
            self.lib.H5Dclose(dset)
            raise RuntimeError(f"ambiguous coords axes for {path}: shape={shape}")

        frame_axis = frame_axes[0]
        atom_axis = atom_axes[0]
        xyz_axis = xyz_axes[0]
        count = [0] * len(shape)
        count[frame_axis] = 1
        count[atom_axis] = num_atoms
        count[xyz_axis] = 3
        count_arr = (c_ulonglong * len(shape))(*count)
        mem_space = self.lib.H5Screate_simple(len(shape), count_arr, None)
        file_space = self.lib.H5Dget_space(dset)
        tmp = np.empty(tuple(count), dtype=np.float32)

        def read_frame(frame_index: int) -> np.ndarray:
            start = [0] * len(shape)
            start[frame_axis] = int(frame_index)
            start_arr = (c_ulonglong * len(shape))(*start)
            status = self.lib.H5Sselect_hyperslab(file_space, 0, start_arr, None, count_arr, None)
            if status < 0:
                raise RuntimeError(f"could not select frame {frame_index} in {path}")
            status = self.lib.H5Dread(
                dset,
                self.h5t_native_float,
                mem_space,
                file_space,
                0,
                tmp.ctypes.data_as(c_void_p),
            )
            if status < 0:
                raise RuntimeError(f"could not read frame {frame_index} in {path}")
            arr = np.take(tmp, 0, axis=frame_axis)
            adjusted_atom_axis = atom_axis - (1 if frame_axis < atom_axis else 0)
            adjusted_xyz_axis = xyz_axis - (1 if frame_axis < xyz_axis else 0)
            return np.moveaxis(arr, (adjusted_atom_axis, adjusted_xyz_axis), (0, 1)).copy()

        def close() -> None:
            self.lib.H5Sclose(file_space)
            self.lib.H5Sclose(mem_space)
            self.lib.H5Dclose(dset)

        return read_frame, close


def read_domains(path: Path) -> list[str]:
    if path.suffix == ".tsv":
        with path.open() as handle:
            return [row["domain"] for row in csv.DictReader(handle, delimiter="\t")]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def atom_lines(pdb_text: str) -> list[str]:
    return [line for line in pdb_text.splitlines() if line.startswith(("ATOM", "HETATM"))]


def format_atom_line(template: str, xyz: np.ndarray) -> str:
    line = template.rstrip("\n")
    if len(line) < 54:
        line = line.ljust(54)
    return f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}\n"


def export_domain(
    h5: H5,
    domain: str,
    data_dir: Path,
    out_dir: Path,
    temperature: str,
    requested_runs: list[str],
    frame_stride: int,
) -> dict:
    h5_path = data_dir / f"mdcath_dataset_{domain}.h5"
    file_id = h5.lib.H5Fopen(str(h5_path).encode(), 0, 0)
    if file_id < 0:
        raise RuntimeError(f"could not open {h5_path}")

    present_runs: list[str] = []
    missing_runs: list[str] = []
    model_count = 0
    frame_count = 0
    num_atoms = None
    out_path = out_dir / f"{domain}_mdcath_{temperature}_runs_multimodel.pdb"

    try:
        domain_group = h5.lib.H5Gopen2(file_id, domain.encode(), 0)
        if domain_group < 0:
            raise RuntimeError(f"could not open domain group {domain}")
        try:
            num_atoms = h5.read_int_attr(domain_group, "numProteinAtoms")
        finally:
            h5.lib.H5Gclose(domain_group)
        if num_atoms is None:
            raise RuntimeError(f"missing numProteinAtoms attr for {domain}")

        pdb_text = h5.read_string(file_id, f"{domain}/pdbProteinAtoms")
        templates = atom_lines(pdb_text)
        if len(templates) != num_atoms:
            raise RuntimeError(f"{domain}: PDB atom lines {len(templates)} != numProteinAtoms {num_atoms}")

        out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as out:
            out.write(f"REMARK 900 MDCATH_DOMAIN {domain}\n")
            out.write(f"REMARK 901 TEMPERATURE {temperature}\n")
            out.write(f"REMARK 902 REQUESTED_RUNS {','.join(requested_runs)}\n")

            for run in requested_runs:
                run_path = f"{domain}/{temperature}/{run}"
                coords_path = f"{run_path}/coords"
                if not h5.exists(file_id, coords_path):
                    missing_runs.append(run)
                    continue
                run_group = h5.lib.H5Gopen2(file_id, run_path.encode(), 0)
                if run_group < 0:
                    missing_runs.append(run)
                    continue
                try:
                    num_frames = h5.read_int_attr(run_group, "numFrames")
                finally:
                    h5.lib.H5Gclose(run_group)
                if num_frames is None:
                    raise RuntimeError(f"{domain} run {run}: missing numFrames")

                present_runs.append(run)
                read_frame, close_reader = h5.make_frame_reader(file_id, coords_path, num_atoms, num_frames)
                try:
                    for frame in range(0, num_frames, frame_stride):
                        coords = read_frame(frame)
                        if coords.shape != (num_atoms, 3):
                            raise RuntimeError(f"{domain} run {run} frame {frame}: bad shape {coords.shape}")
                        model_count += 1
                        frame_count += 1
                        out.write(f"MODEL     {model_count:4d}\n")
                        out.write(f"REMARK 910 DOMAIN {domain} RUN {run} FRAME {frame} TEMPERATURE {temperature}\n")
                        for template, xyz in zip(templates, coords):
                            out.write(format_atom_line(template, xyz))
                        out.write("ENDMDL\n")
                finally:
                    close_reader()

            out.write(f"REMARK 903 PRESENT_RUNS {','.join(present_runs)}\n")
            out.write(f"REMARK 904 MISSING_RUNS {','.join(missing_runs) if missing_runs else 'none'}\n")
            out.write("END\n")
    finally:
        h5.lib.H5Fclose(file_id)

    return {
        "domain": domain,
        "temperature": temperature,
        "requested_runs": ",".join(requested_runs),
        "present_runs": ",".join(present_runs),
        "missing_runs": ",".join(missing_runs),
        "frame_stride": frame_stride,
        "models": model_count,
        "frames_exported": frame_count,
        "num_atoms": int(num_atoms or 0),
        "pdb_path": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=Path, default=Path(__file__).with_name("domains.txt"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "md_trajectory_multimodel_pdbs")
    parser.add_argument("--temperature", default="320")
    parser.add_argument("--runs", nargs="+", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    domains = read_domains(args.domains)
    h5 = H5()
    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        row = export_domain(
            h5=h5,
            domain=domain,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            temperature=args.temperature,
            requested_runs=[str(run) for run in args.runs],
            frame_stride=args.frame_stride,
        )
        rows.append(row)
        print(
            f"wrote {row['pdb_path']} models={row['models']} present_runs={row['present_runs']} missing_runs={row['missing_runs'] or 'none'}",
            flush=True,
        )

    report = args.out_dir / "md_multimodel_export_report.tsv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
