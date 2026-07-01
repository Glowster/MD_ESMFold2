#!/usr/bin/env python3
"""Inspect current-frame x_t C-alpha RBF utilization on mdCATH frames.

This is a standalone diagnostic for the MD-conditioning path in
`modeling_esmfold2.py`. It samples current MD frames, extracts the C-alpha
coordinates that `MDConditioning` uses from the mdCATH PDB atom template, and
reports how the current RBF centers are used.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import ctypes.util
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCAL_TRANSFORMERS = ROOT / "transformers" / "src"

MODEL_FILE = LOCAL_TRANSFORMERS / "transformers" / "models" / "esmfold2" / "modeling_esmfold2.py"
H5P_DEFAULT = 0
H5S_ALL = 0
H5F_ACC_RDONLY = 0
H5_INDEX_NAME = 0
H5_ITER_INC = 0
H5S_SELECT_SET = 0

PROTEIN_1TO3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

PROTEIN_RESNAMES = set(PROTEIN_1TO3.values())

RESIDUE_NAME_ALIASES = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
}


@dataclass
class DomainStats:
    frames: int = 0
    pairs: int = 0
    distances: list[np.ndarray] = field(default_factory=list)
    argmax_counts: np.ndarray | None = None
    active_01_counts: np.ndarray | None = None
    active_001_counts: np.ndarray | None = None
    low_max_01: int = 0
    low_max_001: int = 0


def as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return as_text(value.tolist())
    if isinstance(value, list):
        return "".join(as_text(v) for v in value)
    return str(value)


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


def pdb_residue_groups(pdb_text: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current_key: tuple[str, str, str] | None = None
    current: dict[str, Any] | None = None
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


def find_axis(shape: tuple[int, ...], size: int, name: str) -> int:
    axes = [axis for axis, axis_size in enumerate(shape) if axis_size == size]
    if len(axes) != 1:
        raise ValueError(f"could not identify {name} axis in shape {shape}")
    return axes[0]


def ca_md_atom_indices(sequence: str, pdb_text: str) -> np.ndarray:
    groups = [
        group
        for group in pdb_residue_groups(pdb_text)
        if group["resname"] in PROTEIN_RESNAMES
    ]
    expected_resnames = [PROTEIN_1TO3.get(letter, "UNK") for letter in sequence]
    if expected_resnames and len(groups) != len(expected_resnames):
        raise ValueError(
            f"PDB template has {len(groups)} protein residue groups, "
            f"but sequence has {len(expected_resnames)} residues"
        )

    indices = np.full(len(expected_resnames), -1, dtype=np.int64)
    if not expected_resnames:
        expected_resnames = [group["resname"] for group in groups]
        indices = np.full(len(expected_resnames), -1, dtype=np.int64)

    for token_idx, (expected, group) in enumerate(zip(expected_resnames, groups)):
        if expected != "UNK" and group["resname"] != expected:
            raise ValueError(
                f"Residue {token_idx} is {expected} in sequence but "
                f"{group['resname']} in PDB atom template"
            )
        by_name = {name: md_idx for name, md_idx in group["atoms"]}
        ca_idx = by_name.get("CA")
        if ca_idx is None:
            raise ValueError(f"missing CA for residue {token_idx} {group['resname']}")
        indices[token_idx] = ca_idx
    return indices


def get_ca_frame_rbf_defaults() -> tuple[np.ndarray, float]:
    tree = ast.parse(MODEL_FILE.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "CAFramePairEncoder":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            defaults = item.args.defaults
            args = item.args.args[-len(defaults) :]
            values = {
                arg.arg: ast.literal_eval(default)
                for arg, default in zip(args, defaults)
            }
            num_rbf = int(values["num_rbf"])
            rbf_min = float(values["rbf_min"])
            rbf_max = float(values["rbf_max"])
            rbf_width = float(values["rbf_width"])
            return np.linspace(rbf_min, rbf_max, num_rbf, dtype=np.float32), rbf_width
    raise RuntimeError(f"could not find CAFramePairEncoder defaults in {MODEL_FILE}")


class CTypesH5File:
    """Tiny read-only HDF5 wrapper for the mdCATH fields used here."""

    hid_t = ctypes.c_longlong
    hsize_t = ctypes.c_ulonglong
    herr_t = ctypes.c_int
    _callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int,
        hid_t,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lib = self._load_library()
        self._configure_library()
        self.file_id = self._check_id(
            self.lib.H5Fopen(str(path).encode(), H5F_ACC_RDONLY, H5P_DEFAULT),
            f"H5Fopen({path})",
        )

    @staticmethod
    def _load_library():
        name = ctypes.util.find_library("hdf5_serial") or "libhdf5_serial.so.103"
        return ctypes.CDLL(name)

    def _configure_library(self) -> None:
        hid_t = self.hid_t
        hsize_t = self.hsize_t

        self.lib.H5open.restype = ctypes.c_int
        self.lib.H5open.argtypes = []
        self.lib.H5open()

        self.lib.H5Fopen.restype = hid_t
        self.lib.H5Fopen.argtypes = [ctypes.c_char_p, ctypes.c_uint, hid_t]
        self.lib.H5Fclose.restype = ctypes.c_int
        self.lib.H5Fclose.argtypes = [hid_t]

        self.lib.H5Gopen2.restype = hid_t
        self.lib.H5Gopen2.argtypes = [hid_t, ctypes.c_char_p, hid_t]
        self.lib.H5Gclose.restype = ctypes.c_int
        self.lib.H5Gclose.argtypes = [hid_t]

        self.lib.H5Dopen2.restype = hid_t
        self.lib.H5Dopen2.argtypes = [hid_t, ctypes.c_char_p, hid_t]
        self.lib.H5Dclose.restype = ctypes.c_int
        self.lib.H5Dclose.argtypes = [hid_t]
        self.lib.H5Dget_type.restype = hid_t
        self.lib.H5Dget_type.argtypes = [hid_t]
        self.lib.H5Dget_space.restype = hid_t
        self.lib.H5Dget_space.argtypes = [hid_t]
        self.lib.H5Dread.restype = ctypes.c_int
        self.lib.H5Dread.argtypes = [
            hid_t,
            hid_t,
            hid_t,
            hid_t,
            hid_t,
            ctypes.c_void_p,
        ]

        self.lib.H5Aexists.restype = ctypes.c_int
        self.lib.H5Aexists.argtypes = [hid_t, ctypes.c_char_p]
        self.lib.H5Aopen.restype = hid_t
        self.lib.H5Aopen.argtypes = [hid_t, ctypes.c_char_p, hid_t]
        self.lib.H5Aclose.restype = ctypes.c_int
        self.lib.H5Aclose.argtypes = [hid_t]
        self.lib.H5Aget_type.restype = hid_t
        self.lib.H5Aget_type.argtypes = [hid_t]
        self.lib.H5Aread.restype = ctypes.c_int
        self.lib.H5Aread.argtypes = [hid_t, hid_t, ctypes.c_void_p]

        self.lib.H5Tget_size.restype = ctypes.c_size_t
        self.lib.H5Tget_size.argtypes = [hid_t]
        self.lib.H5Tis_variable_str.restype = ctypes.c_int
        self.lib.H5Tis_variable_str.argtypes = [hid_t]
        self.lib.H5Tclose.restype = ctypes.c_int
        self.lib.H5Tclose.argtypes = [hid_t]

        self.lib.H5Sget_simple_extent_ndims.restype = ctypes.c_int
        self.lib.H5Sget_simple_extent_ndims.argtypes = [hid_t]
        self.lib.H5Sget_simple_extent_dims.restype = ctypes.c_int
        self.lib.H5Sget_simple_extent_dims.argtypes = [
            hid_t,
            ctypes.POINTER(hsize_t),
            ctypes.POINTER(hsize_t),
        ]
        self.lib.H5Screate_simple.restype = hid_t
        self.lib.H5Screate_simple.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(hsize_t),
            ctypes.POINTER(hsize_t),
        ]
        self.lib.H5Sselect_hyperslab.restype = ctypes.c_int
        self.lib.H5Sselect_hyperslab.argtypes = [
            hid_t,
            ctypes.c_int,
            ctypes.POINTER(hsize_t),
            ctypes.POINTER(hsize_t),
            ctypes.POINTER(hsize_t),
            ctypes.POINTER(hsize_t),
        ]
        self.lib.H5Sclose.restype = ctypes.c_int
        self.lib.H5Sclose.argtypes = [hid_t]

        self.lib.H5Literate.restype = ctypes.c_int
        self.lib.H5Literate.argtypes = [
            hid_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(hsize_t),
            self._callback_type,
            ctypes.c_void_p,
        ]

        self.lib.H5free_memory.restype = ctypes.c_int
        self.lib.H5free_memory.argtypes = [ctypes.c_void_p]

        self.H5T_NATIVE_FLOAT = hid_t.in_dll(self.lib, "H5T_NATIVE_FLOAT_g").value
        self.H5T_NATIVE_LLONG = hid_t.in_dll(self.lib, "H5T_NATIVE_LLONG_g").value

    @staticmethod
    def _check_id(value: int, label: str) -> int:
        if value < 0:
            raise RuntimeError(f"{label} failed")
        return int(value)

    @staticmethod
    def _check_err(value: int, label: str) -> None:
        if value < 0:
            raise RuntimeError(f"{label} failed")

    def close(self) -> None:
        if getattr(self, "file_id", -1) >= 0:
            self.lib.H5Fclose(self.file_id)
            self.file_id = -1

    def __enter__(self) -> "CTypesH5File":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open_group(self, loc_id: int, name: str) -> int:
        return self._check_id(
            self.lib.H5Gopen2(loc_id, name.encode(), H5P_DEFAULT),
            f"H5Gopen2({name})",
        )

    def close_group(self, group_id: int) -> None:
        self.lib.H5Gclose(group_id)

    def open_dataset(self, loc_id: int, name: str) -> int:
        return self._check_id(
            self.lib.H5Dopen2(loc_id, name.encode(), H5P_DEFAULT),
            f"H5Dopen2({name})",
        )

    def close_dataset(self, dataset_id: int) -> None:
        self.lib.H5Dclose(dataset_id)

    def list_names(self, loc_id: int) -> list[str]:
        names: list[str] = []

        def visitor(_group_id, name, _info, _op_data):
            names.append(name.decode())
            return 0

        callback = self._callback_type(visitor)
        idx = self.hsize_t(0)
        self._check_err(
            self.lib.H5Literate(
                loc_id,
                H5_INDEX_NAME,
                H5_ITER_INC,
                ctypes.byref(idx),
                callback,
                None,
            ),
            "H5Literate",
        )
        return names

    def dataset_shape(self, dataset_id: int) -> tuple[int, ...]:
        space_id = self._check_id(
            self.lib.H5Dget_space(dataset_id),
            "H5Dget_space",
        )
        try:
            rank = self.lib.H5Sget_simple_extent_ndims(space_id)
            if rank < 0:
                raise RuntimeError("H5Sget_simple_extent_ndims failed")
            dims = (self.hsize_t * rank)()
            self._check_err(
                self.lib.H5Sget_simple_extent_dims(space_id, dims, None),
                "H5Sget_simple_extent_dims",
            )
            return tuple(int(dim) for dim in dims)
        finally:
            self.lib.H5Sclose(space_id)

    def read_string_dataset(self, loc_id: int, name: str) -> str:
        dataset_id = self.open_dataset(loc_id, name)
        try:
            dtype_id = self._check_id(self.lib.H5Dget_type(dataset_id), "H5Dget_type")
            space_id = self._check_id(self.lib.H5Dget_space(dataset_id), "H5Dget_space")
            try:
                return self._read_string(dataset_id, dtype_id, space_id, is_attr=False)
            finally:
                self.lib.H5Sclose(space_id)
                self.lib.H5Tclose(dtype_id)
        finally:
            self.close_dataset(dataset_id)

    def read_int_attr(self, loc_id: int, name: str, default: int | None = None) -> int:
        exists = self.lib.H5Aexists(loc_id, name.encode())
        if exists == 0 and default is not None:
            return int(default)
        if exists <= 0:
            raise RuntimeError(f"attribute {name!r} is missing")
        attr_id = self._check_id(
            self.lib.H5Aopen(loc_id, name.encode(), H5P_DEFAULT),
            f"H5Aopen({name})",
        )
        try:
            value = ctypes.c_longlong()
            self._check_err(
                self.lib.H5Aread(
                    attr_id,
                    self.H5T_NATIVE_LLONG,
                    ctypes.byref(value),
                ),
                f"H5Aread({name})",
            )
            return int(value.value)
        finally:
            self.lib.H5Aclose(attr_id)

    def _read_string(
        self,
        object_id: int,
        dtype_id: int,
        space_id: int,
        is_attr: bool,
    ) -> str:
        if self.lib.H5Tis_variable_str(dtype_id) > 0:
            value = ctypes.c_void_p()
            read = self.lib.H5Aread if is_attr else self.lib.H5Dread
            if is_attr:
                self._check_err(read(object_id, dtype_id, ctypes.byref(value)), "H5Aread string")
            else:
                self._check_err(
                    read(
                        object_id,
                        dtype_id,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        ctypes.byref(value),
                    ),
                    "H5Dread string",
                )
            text = ctypes.string_at(value.value).decode("utf-8") if value.value else ""
            if value.value:
                self.lib.H5free_memory(value)
            return text

        size = int(self.lib.H5Tget_size(dtype_id))
        if size <= 0:
            return ""
        buffer = ctypes.create_string_buffer(size + 1)
        if is_attr:
            self._check_err(
                self.lib.H5Aread(object_id, dtype_id, buffer),
                "H5Aread fixed string",
            )
        else:
            self._check_err(
                self.lib.H5Dread(
                    object_id,
                    dtype_id,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    buffer,
                ),
                "H5Dread fixed string",
            )
        return buffer.raw.rstrip(b"\x00").decode("utf-8")

    def read_frame(
        self,
        dataset_id: int,
        num_atoms: int,
        num_frames: int,
        frame: int,
    ) -> np.ndarray:
        shape = self.dataset_shape(dataset_id)
        xyz_axis = find_axis(shape, 3, "xyz")
        atom_axis = find_axis(shape, num_atoms, "atom")
        frame_axis = find_axis(shape, num_frames, "frame")

        rank = len(shape)
        start_values = [0] * rank
        count_values = list(shape)
        start_values[frame_axis] = int(frame)
        count_values[frame_axis] = 1

        start = (self.hsize_t * rank)(*start_values)
        count = (self.hsize_t * rank)(*count_values)
        file_space = self._check_id(self.lib.H5Dget_space(dataset_id), "H5Dget_space")
        mem_space = self._check_id(
            self.lib.H5Screate_simple(rank, count, None),
            "H5Screate_simple",
        )
        try:
            self._check_err(
                self.lib.H5Sselect_hyperslab(
                    file_space,
                    H5S_SELECT_SET,
                    start,
                    None,
                    count,
                    None,
                ),
                "H5Sselect_hyperslab",
            )
            out = np.empty(int(np.prod(count_values)), dtype=np.float32)
            self._check_err(
                self.lib.H5Dread(
                    dataset_id,
                    self.H5T_NATIVE_FLOAT,
                    mem_space,
                    file_space,
                    H5P_DEFAULT,
                    out.ctypes.data_as(ctypes.c_void_p),
                ),
                "H5Dread frame",
            )
        finally:
            self.lib.H5Sclose(mem_space)
            self.lib.H5Sclose(file_space)

        data = out.reshape(tuple(count_values))
        data = np.take(data, 0, axis=frame_axis)
        adjusted_atom_axis = atom_axis - (1 if frame_axis < atom_axis else 0)
        adjusted_xyz_axis = xyz_axis - (1 if frame_axis < xyz_axis else 0)
        data = np.moveaxis(data, (adjusted_atom_axis, adjusted_xyz_axis), (0, 1))
        return data.astype("float32", copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "mdcath_320K_len_le200" / "data",
    )
    parser.add_argument("--temperature", default="320")
    parser.add_argument("--sample-frames", type=int, default=1000)
    parser.add_argument("--max-domains", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-diagonal", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--print-bins", action="store_true")
    parser.add_argument("--rbf-active-threshold", type=float, default=0.01)
    return parser.parse_args()


def domain_from_path(path: Path) -> str:
    prefix = "mdcath_dataset_"
    name = path.stem
    return name[len(prefix) :] if name.startswith(prefix) else name


def read_one_frame(coords, num_atoms: int, num_frames: int, frame: int) -> np.ndarray:
    xyz_axis = find_axis(coords.shape, 3, "xyz")
    atom_axis = find_axis(coords.shape, num_atoms, "atom")
    frame_axis = find_axis(coords.shape, num_frames, "frame")

    selection = [slice(None)] * coords.ndim
    selection[frame_axis] = int(frame)
    data = coords[tuple(selection)]

    adjusted_atom_axis = atom_axis - (1 if frame_axis < atom_axis else 0)
    adjusted_xyz_axis = xyz_axis - (1 if frame_axis < xyz_axis else 0)
    data = np.moveaxis(data, (adjusted_atom_axis, adjusted_xyz_axis), (0, 1))
    return data.astype("float32", copy=False)


def collect_records(data_dir: Path, temperature: str) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError:
        return collect_records_ctypes(data_dir, temperature)

    records: list[dict[str, Any]] = []
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
                if g["resname"] in PROTEIN_RESNAMES
            ]
            if len(protein_groups) != len(sequence):
                skipped_unmappable += 1
                continue
            temperatures = [key for key in group.keys() if key.isdigit()]
            if temperature not in temperatures:
                continue
            run_frames = {}
            for run_key in sorted(group[temperature].keys(), key=int):
                run_group = group[temperature][run_key]
                coords = run_group["coords"]
                num_frames = int(run_group.attrs.get("numFrames", coords.shape[-1]))
                if num_frames > 0:
                    run_frames[run_key] = num_frames
            if not run_frames:
                continue
            records.append(
                {
                    "path": path,
                    "domain": domain,
                    "sequence": sequence,
                    "length": len(sequence),
                    "run_frames": run_frames,
                }
            )
    if not records:
        raise FileNotFoundError(f"no usable mdCATH records found under {data_dir}")
    if skipped_unmappable:
        print(f"skipped_unmappable_domains={skipped_unmappable}")
    return records


def collect_records_ctypes(data_dir: Path, temperature: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped_unmappable = 0
    for path in sorted(data_dir.glob("mdcath_dataset_*.h5")):
        domain = domain_from_path(path)
        with CTypesH5File(path) as h5:
            group_id = h5.open_group(h5.file_id, domain)
            try:
                sequence = h5.read_string_dataset(group_id, "sequence")
                pdb_text = h5.read_string_dataset(group_id, "pdbProteinAtoms")
                protein_groups = [
                    g
                    for g in pdb_residue_groups(pdb_text)
                    if g["resname"] in PROTEIN_RESNAMES
                ]
                if len(protein_groups) != len(sequence):
                    skipped_unmappable += 1
                    continue
                if temperature not in h5.list_names(group_id):
                    continue
                temp_id = h5.open_group(group_id, temperature)
                try:
                    run_frames = {}
                    for run_key in sorted(h5.list_names(temp_id), key=int):
                        run_id = h5.open_group(temp_id, run_key)
                        try:
                            coords_id = h5.open_dataset(run_id, "coords")
                            try:
                                shape = h5.dataset_shape(coords_id)
                                num_frames = h5.read_int_attr(
                                    run_id,
                                    "numFrames",
                                    default=shape[-1],
                                )
                            finally:
                                h5.close_dataset(coords_id)
                            if num_frames > 0:
                                run_frames[run_key] = num_frames
                        finally:
                            h5.close_group(run_id)
                    if not run_frames:
                        continue
                    records.append(
                        {
                            "path": path,
                            "domain": domain,
                            "sequence": sequence,
                            "length": len(protein_groups),
                            "run_frames": run_frames,
                        }
                    )
                finally:
                    h5.close_group(temp_id)
            finally:
                h5.close_group(group_id)
    if not records:
        raise FileNotFoundError(f"no usable mdCATH records found under {data_dir}")
    if skipped_unmappable:
        print(f"skipped_unmappable_domains={skipped_unmappable}")
    return records


def frame_ca_coords(
    record: dict[str, Any],
    temperature: str,
    run_key: str,
    frame: int,
) -> np.ndarray:
    try:
        import h5py
    except ImportError:
        return frame_ca_coords_ctypes(record, temperature, run_key, frame)

    path = record["path"]
    domain = record["domain"]
    sequence = record["sequence"]
    with h5py.File(path, "r") as h5:
        group = h5[domain]
        run_group = group[temperature][run_key]
        coords = run_group["coords"]
        num_atoms = int(group.attrs["numProteinAtoms"])
        num_frames = int(run_group.attrs.get("numFrames", coords.shape[-1]))
        pdb_text = as_text(group["pdbProteinAtoms"][()])
        md_coords = read_one_frame(coords, num_atoms, num_frames, frame)

    ca_indices = ca_md_atom_indices(sequence, pdb_text)
    if ca_indices.size and int(ca_indices.max()) >= md_coords.shape[0]:
        raise ValueError(f"invalid CA mapping for {domain}")
    return md_coords[ca_indices]


def frame_ca_coords_ctypes(
    record: dict[str, Any],
    temperature: str,
    run_key: str,
    frame: int,
) -> np.ndarray:
    path = record["path"]
    domain = record["domain"]
    sequence = record["sequence"]
    with CTypesH5File(path) as h5:
        group_id = h5.open_group(h5.file_id, domain)
        try:
            temp_id = h5.open_group(group_id, temperature)
            try:
                run_id = h5.open_group(temp_id, run_key)
                try:
                    coords_id = h5.open_dataset(run_id, "coords")
                    try:
                        num_atoms = h5.read_int_attr(group_id, "numProteinAtoms")
                        shape = h5.dataset_shape(coords_id)
                        num_frames = h5.read_int_attr(
                            run_id,
                            "numFrames",
                            default=shape[-1],
                        )
                        pdb_text = h5.read_string_dataset(group_id, "pdbProteinAtoms")
                        md_coords = h5.read_frame(coords_id, num_atoms, num_frames, frame)
                    finally:
                        h5.close_dataset(coords_id)
                finally:
                    h5.close_group(run_id)
            finally:
                h5.close_group(temp_id)
        finally:
            h5.close_group(group_id)

    ca_indices = ca_md_atom_indices(sequence, pdb_text)
    if ca_indices.size and int(ca_indices.max()) >= md_coords.shape[0]:
        raise ValueError(f"invalid CA mapping for {domain}")
    return md_coords[ca_indices]


def pair_distances(ca_xyz: np.ndarray, include_diagonal: bool) -> np.ndarray:
    diff = ca_xyz[:, None, :] - ca_xyz[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32), dtype=np.float32)
    if include_diagonal:
        return distances.reshape(-1)
    mask = ~np.eye(distances.shape[0], dtype=bool)
    return distances[mask]


def update_stats(
    stats: DomainStats,
    distances: np.ndarray,
    centers: np.ndarray,
    width: float,
) -> None:
    if stats.argmax_counts is None:
        stats.argmax_counts = np.zeros_like(centers, dtype=np.int64)
        stats.active_01_counts = np.zeros_like(centers, dtype=np.int64)
        stats.active_001_counts = np.zeros_like(centers, dtype=np.int64)

    rbf = np.exp(-np.square((distances[:, None] - centers[None, :]) / width))
    max_values = rbf.max(axis=1)
    argmax = rbf.argmax(axis=1)

    stats.frames += 1
    stats.pairs += int(distances.size)
    stats.distances.append(distances.astype(np.float32, copy=False))
    stats.argmax_counts += np.bincount(argmax, minlength=centers.size)
    assert stats.active_01_counts is not None
    assert stats.active_001_counts is not None
    stats.active_01_counts += (rbf > 0.1).sum(axis=0)
    stats.active_001_counts += (rbf > 0.01).sum(axis=0)
    stats.low_max_01 += int((max_values < 0.1).sum())
    stats.low_max_001 += int((max_values < 0.01).sum())


def quantiles(values: np.ndarray) -> dict[str, float]:
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    result = np.percentile(values, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, result)}


def summarize_domain(domain: str, stats: DomainStats) -> dict[str, Any]:
    distances = np.concatenate(stats.distances)
    assert stats.argmax_counts is not None
    assert stats.active_01_counts is not None
    assert stats.active_001_counts is not None
    pairs = max(stats.pairs, 1)
    return {
        "domain": domain,
        "frames": stats.frames,
        "pairs": stats.pairs,
        "distance_quantiles": quantiles(distances),
        "argmax_fraction": (stats.argmax_counts / pairs).tolist(),
        "active_fraction_gt_0_1": (stats.active_01_counts / pairs).tolist(),
        "active_fraction_gt_0_01": (stats.active_001_counts / pairs).tolist(),
        "fraction_pairs_with_max_rbf_lt_0_1": stats.low_max_01 / pairs,
        "fraction_pairs_with_max_rbf_lt_0_01": stats.low_max_001 / pairs,
    }


def print_bin_table(
    centers: np.ndarray,
    argmax_fraction: np.ndarray,
    active_fraction: np.ndarray,
) -> None:
    print("bin\tcenter_A\targmax_frac\tactive_frac_gt_0.01")
    for idx, (center, argmax, active) in enumerate(
        zip(centers, argmax_fraction, active_fraction)
    ):
        print(f"{idx:02d}\t{center:6.2f}\t{argmax:11.6f}\t{active:18.6f}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    centers, width = get_ca_frame_rbf_defaults()

    records = collect_records(args.data_dir, args.temperature)
    rng.shuffle(records)
    records = records[: min(args.max_domains, len(records))]

    assignments = []
    for idx in range(args.sample_frames):
        record = records[idx % len(records)]
        assignments.append(record)
    rng.shuffle(assignments)

    global_stats = DomainStats()
    by_domain: dict[str, DomainStats] = {}
    failures: list[str] = []

    for record in assignments:
        domain = record["domain"]
        try:
            run_key, num_frames = rng.choice(list(record["run_frames"].items()))
            frame = rng.randrange(int(num_frames))
            ca_xyz = frame_ca_coords(record, args.temperature, run_key, frame)
            distances = pair_distances(ca_xyz, args.include_diagonal)
            update_stats(global_stats, distances, centers, width)
            update_stats(by_domain.setdefault(domain, DomainStats()), distances, centers, width)
        except Exception as err:  # noqa: BLE001 - diagnostic should continue.
            failures.append(f"{domain}: {err}")

    if global_stats.frames == 0:
        raise RuntimeError(f"all sampled frames failed: {failures[:5]}")

    global_summary = summarize_domain("ALL", global_stats)
    domain_summaries = [
        summarize_domain(domain, stats)
        for domain, stats in sorted(by_domain.items())
        if stats.frames > 0
    ]
    pairs = max(global_stats.pairs, 1)
    assert global_stats.argmax_counts is not None
    assert global_stats.active_001_counts is not None
    argmax_fraction = global_stats.argmax_counts / pairs
    active_fraction = global_stats.active_001_counts / pairs

    dead_argmax_bins = np.flatnonzero(argmax_fraction < 1e-4).tolist()
    low_active_bins = np.flatnonzero(active_fraction < args.rbf_active_threshold).tolist()

    result = {
        "sampled_frames": global_stats.frames,
        "sampled_domains": len(domain_summaries),
        "failed_frames": len(failures),
        "temperature": args.temperature,
        "include_diagonal": args.include_diagonal,
        "rbf_centers": centers.tolist(),
        "rbf_width": width,
        "global": global_summary,
        "dead_argmax_bins_lt_1e-4": dead_argmax_bins,
        f"low_active_bins_lt_{args.rbf_active_threshold:g}": low_active_bins,
        "domains": domain_summaries,
        "failures_head": failures[:20],
    }

    q = global_summary["distance_quantiles"]
    print(
        "sampled_frames={sampled_frames} sampled_domains={sampled_domains} "
        "pairs={pairs} failed_frames={failed_frames}".format(
            sampled_frames=result["sampled_frames"],
            sampled_domains=result["sampled_domains"],
            pairs=global_summary["pairs"],
            failed_frames=result["failed_frames"],
        )
    )
    print(f"rbf_centers_A={centers[0]:.2f}..{centers[-1]:.2f} n={centers.size} width={width:.2f}")
    print(
        "distance_A "
        f"p1={q['p1']:.2f} p5={q['p5']:.2f} p25={q['p25']:.2f} "
        f"p50={q['p50']:.2f} p75={q['p75']:.2f} p95={q['p95']:.2f} "
        f"p99={q['p99']:.2f} max={q['p100']:.2f}"
    )
    print(
        "max_rbf_low_fraction "
        f"lt_0.1={global_summary['fraction_pairs_with_max_rbf_lt_0_1']:.6f} "
        f"lt_0.01={global_summary['fraction_pairs_with_max_rbf_lt_0_01']:.6f}"
    )
    print(
        f"dead_argmax_bins_lt_1e-4={dead_argmax_bins} "
        f"low_active_bins_lt_{args.rbf_active_threshold:g}={low_active_bins}"
    )

    worst_domains = sorted(
        domain_summaries,
        key=lambda item: item["fraction_pairs_with_max_rbf_lt_0_01"],
        reverse=True,
    )[:10]
    print("worst_domains_by_fraction_pairs_with_max_rbf_lt_0.01")
    for item in worst_domains:
        print(
            f"{item['domain']}\tframes={item['frames']}\t"
            f"pairs={item['pairs']}\t"
            f"low_max_lt_0.01={item['fraction_pairs_with_max_rbf_lt_0_01']:.6f}\t"
            f"p95={item['distance_quantiles']['p95']:.2f}\t"
            f"p99={item['distance_quantiles']['p99']:.2f}\t"
            f"max={item['distance_quantiles']['p100']:.2f}"
        )

    if args.print_bins:
        print_bin_table(centers, argmax_fraction, active_fraction)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote_json={args.out_json}")


if __name__ == "__main__":
    main()
