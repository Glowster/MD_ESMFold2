#!/usr/bin/env python3
"""Export one mdCATH domain HDF5 file to multi-model PDB trajectories.

Usage:
    python export_domain_pdb_trajectories.py 12asA00

Output:
    12asA00/temprature_320/replica_0.pdb
    12asA00/temprature_320/replica_1.pdb
    ...
"""

from pathlib import Path
import argparse
import sys


def as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def atom_template_lines(pdb_text):
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            lines.append(line)
    return lines


def with_xyz(line, x, y, z):
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}\n"


def write_multimodel_pdb(out_path, atom_lines, coords):
    # coords is stored as (xyz, atom, frame)
    with out_path.open("w") as out:
        for frame in range(coords.shape[2]):
            out.write(f"MODEL     {frame + 1:4d}\n")
            for atom, line in enumerate(atom_lines):
                x, y, z = coords[:, atom, frame]
                out.write(with_xyz(line, x, y, z))
            out.write("ENDMDL\n")
        out.write("END\n")


def export_domain(domain, data_dir, out_dir):
    try:
        import h5py
    except ImportError as err:
        raise RuntimeError("This script needs h5py: python -m pip install h5py") from err

    h5_path = data_dir / f"mdcath_dataset_{domain}.h5"
    hd5_path = data_dir / f"mdcath_dataset_{domain}.hd5"
    if not h5_path.exists() and hd5_path.exists():
        h5_path = hd5_path
    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find {h5_path} or {hd5_path}")

    with h5py.File(h5_path, "r") as h5:
        group = h5[domain]
        atom_lines = atom_template_lines(as_text(group["pdbProteinAtoms"][()]))
        num_atoms = int(group.attrs["numProteinAtoms"])

        if len(atom_lines) != num_atoms:
            raise ValueError(f"PDB template has {len(atom_lines)} atoms, expected {num_atoms}")

        domain_dir = out_dir / domain
        for temperature in sorted(k for k in group.keys() if k.isdigit()):
            temperature_dir = domain_dir / f"temprature_{temperature}"
            temperature_dir.mkdir(parents=True, exist_ok=True)

            replicas = sorted(group[temperature].keys(), key=int)
            for replica in replicas:
                coords = group[temperature][replica]["coords"][()]
                out_path = temperature_dir / f"replica_{replica}.pdb"
                write_multimodel_pdb(out_path, atom_lines, coords)
                print(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", help="Domain id, e.g. 12asA00")
    parser.add_argument("--data-dir", type=Path, default=Path("mdcath_minimal_h5/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    try:
        export_domain(args.domain, args.data_dir, args.out_dir)
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
