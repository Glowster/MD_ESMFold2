#!/usr/bin/env python3
"""Compare local filtered mdCATH files with raw Hugging Face mdCATH files."""

from __future__ import annotations

import csv
import hashlib
from ctypes import CDLL, byref, c_char_p, c_int, c_longlong, c_size_t, c_uint, c_void_p, create_string_buffer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "verify_data" / "target_domains.txt"
LOCAL_DATA = ROOT / "data" / "mdcath_320K_len_le200" / "data"
HF_DATA = ROOT / "verify_data" / "hf_raw" / "data"
HF_SOURCE = ROOT / "verify_data" / "hf_raw" / "mdcath_source.h5"
OUT_TSV = ROOT / "verify_data" / "comparison.tsv"

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
}

HISTIDINE_ALIASES = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
}


class H5:
    def __init__(self) -> None:
        self.lib = CDLL("/lib/x86_64-linux-gnu/libhdf5_serial.so.103")
        hid_t = c_longlong
        specs = [
            ("H5open", c_int, []),
            ("H5Fopen", hid_t, [c_char_p, c_uint, hid_t]),
            ("H5Gopen2", hid_t, [hid_t, c_char_p, hid_t]),
            ("H5Dopen2", hid_t, [hid_t, c_char_p, hid_t]),
            ("H5Dget_type", hid_t, [hid_t]),
            ("H5Lexists", c_int, [hid_t, c_char_p, hid_t]),
            ("H5Tget_size", c_size_t, [hid_t]),
            ("H5Tis_variable_str", c_int, [hid_t]),
            ("H5Dread", c_int, [hid_t, hid_t, hid_t, hid_t, hid_t, c_void_p]),
            ("H5Tclose", c_int, [hid_t]),
            ("H5Dclose", c_int, [hid_t]),
            ("H5Aopen", hid_t, [hid_t, c_char_p, hid_t]),
            ("H5Aread", c_int, [hid_t, hid_t, c_void_p]),
            ("H5Aclose", c_int, [hid_t]),
            ("H5Gclose", c_int, [hid_t]),
            ("H5Fclose", c_int, [hid_t]),
        ]
        for name, restype, argtypes in specs:
            fn = getattr(self.lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
        self.lib.H5open()
        self.h5p_default = 0
        self.h5s_all = 0
        self.h5t_native_llong = c_longlong.in_dll(self.lib, "H5T_NATIVE_LLONG_g").value

    def read_string(self, group_id: int, name: str) -> str:
        dset = self.lib.H5Dopen2(group_id, name.encode(), self.h5p_default)
        if dset < 0:
            raise RuntimeError(f"could not open dataset {name}")
        dtype = self.lib.H5Dget_type(dset)
        if dtype < 0:
            raise RuntimeError(f"could not read dtype for {name}")

        if self.lib.H5Tis_variable_str(dtype) > 0:
            ptr = c_char_p()
            status = self.lib.H5Dread(dset, dtype, self.h5s_all, self.h5s_all, self.h5p_default, byref(ptr))
            if status < 0:
                raise RuntimeError(f"could not read variable string {name}")
            data = ptr.value or b""
        else:
            size = self.lib.H5Tget_size(dtype)
            buf = create_string_buffer(size)
            status = self.lib.H5Dread(dset, dtype, self.h5s_all, self.h5s_all, self.h5p_default, buf)
            if status < 0:
                raise RuntimeError(f"could not read fixed string {name}")
            data = buf.raw.rstrip(b"\x00")

        self.lib.H5Tclose(dtype)
        self.lib.H5Dclose(dset)
        return data.decode("utf-8", "replace")

    def read_string_optional(self, group_id: int, name: str) -> str | None:
        if self.lib.H5Lexists(group_id, name.encode(), self.h5p_default) <= 0:
            return None
        return self.read_string(group_id, name)

    def read_int_attr_optional(self, group_id: int, name: str) -> int | None:
        attr = self.lib.H5Aopen(group_id, name.encode(), self.h5p_default)
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

    def read_domain(self, path: Path, domain: str) -> dict:
        file_id = self.lib.H5Fopen(str(path).encode(), 0, self.h5p_default)
        if file_id < 0:
            raise RuntimeError(f"could not open {path}")
        group_id = self.lib.H5Gopen2(file_id, domain.encode(), self.h5p_default)
        if group_id < 0:
            self.lib.H5Fclose(file_id)
            raise RuntimeError(f"could not open group {domain} in {path}")
        try:
            sequence = self.read_string_optional(group_id, "sequence")
            pdb = self.read_string_optional(group_id, "pdbProteinAtoms")
            num_residues = self.read_int_attr_optional(group_id, "numResidues")
            num_protein_atoms = self.read_int_attr_optional(group_id, "numProteinAtoms")
        finally:
            self.lib.H5Gclose(group_id)
            self.lib.H5Fclose(file_id)
        return {
            "path": str(path),
            "sequence": sequence.strip() if sequence is not None else None,
            "pdb": pdb,
            "numResidues": num_residues,
            "numProteinAtoms": num_protein_atoms,
        }


def residue_name(line: str) -> str:
    name = line[17:20].strip()
    return HISTIDINE_ALIASES.get(name, name)


def residue_key(line: str) -> tuple[str, str, str]:
    return (line[21].strip(), line[22:26].strip(), line[26].strip())


def pdb_protein_residues(pdb: str) -> list[tuple[tuple[str, str, str], str, str]]:
    residues = []
    last_key = None
    for line in pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = residue_key(line)
        name = residue_name(line)
        if key != last_key and name in AA3_TO_1:
            residues.append((key, name, AA3_TO_1[name]))
            last_key = key
    return residues


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarize(h5: H5, data_dir: Path, domain: str) -> dict:
    path = data_dir / f"mdcath_dataset_{domain}.h5"
    summary = h5.read_domain(path, domain)
    sequence = summary["sequence"]
    pdb = summary["pdb"]
    residues = pdb_protein_residues(pdb) if pdb is not None else []
    return {
        "path": summary["path"],
        "sequence": sequence,
        "pdb": pdb,
        "sequence_len": len(sequence) if sequence is not None else None,
        "pdb_protein_residue_count": len(residues),
        "numResidues_attr": summary["numResidues"],
        "numProteinAtoms_attr": summary["numProteinAtoms"],
        "sequence_sha256": digest(sequence) if sequence is not None else "",
        "pdbProteinAtoms_sha256": digest(pdb) if pdb is not None else "",
        "pdb_residue_sequence": "".join(r[2] for r in residues),
    }


def summarize_source(h5: H5, domain: str) -> dict:
    summary = h5.read_domain(HF_SOURCE, domain)
    return {
        "path": summary["path"],
        "numResidues_attr": summary["numResidues"],
        "numProteinAtoms_attr": summary["numProteinAtoms"],
    }


def main() -> int:
    domains = [line.strip() for line in TARGETS.read_text().splitlines() if line.strip()]
    h5 = H5()
    rows = []
    for domain in domains:
        local = summarize(h5, LOCAL_DATA, domain)
        hf = summarize(h5, HF_DATA, domain)
        source = summarize_source(h5, domain)
        local_sequence_len = local["sequence_len"]
        hf_num_residues = hf["numResidues_attr"]
        rows.append(
            {
                "domain": domain,
                "local_sequence_len": local_sequence_len,
                "hf_sequence_len": hf["sequence_len"],
                "local_numResidues_attr": local["numResidues_attr"],
                "hf_numResidues_attr": hf_num_residues,
                "source_numResidues_attr": source["numResidues_attr"],
                "local_numProteinAtoms_attr": local["numProteinAtoms_attr"],
                "hf_numProteinAtoms_attr": hf["numProteinAtoms_attr"],
                "local_pdb_residue_count": local["pdb_protein_residue_count"],
                "hf_pdb_residue_count": hf["pdb_protein_residue_count"],
                "hf_has_sequence": hf["sequence"] is not None,
                "sequence_equal": (local["sequence"] == hf["sequence"]) if hf["sequence"] is not None else "",
                "pdbProteinAtoms_equal": local["pdb"] == hf["pdb"],
                "local_sequence_vs_pdb_delta": local["pdb_protein_residue_count"] - local_sequence_len,
                "local_sequence_vs_hf_numResidues_delta": hf_num_residues - local_sequence_len,
                "hf_numResidues_vs_pdb_delta": hf["pdb_protein_residue_count"] - hf_num_residues,
                "local_sequence_sha256": local["sequence_sha256"],
                "hf_sequence_sha256": hf["sequence_sha256"],
                "local_pdbProteinAtoms_sha256": local["pdbProteinAtoms_sha256"],
                "hf_pdbProteinAtoms_sha256": hf["pdbProteinAtoms_sha256"],
            }
        )

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_TSV}")
    for row in rows:
        print(
            row["domain"],
            f"hf_has_sequence={row['hf_has_sequence']}",
            f"pdb_equal={row['pdbProteinAtoms_equal']}",
            f"local_seq_len={row['local_sequence_len']}",
            f"hf_numResidues={row['hf_numResidues_attr']}",
            f"delta={row['local_sequence_vs_hf_numResidues_delta']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
