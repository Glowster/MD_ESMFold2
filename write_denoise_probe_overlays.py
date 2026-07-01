#!/usr/bin/env python3
"""Create flat denoise probe overlay/noisy PDBs from legacy sigma folders."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="run directory containing denoise_probe; default is latest runs/md_esmfold2_*",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def latest_run_dir() -> Path:
    candidates = sorted((ROOT / "runs").glob("md_esmfold2_*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("no runs/md_esmfold2_* directories found")
    return candidates[-1]


def atom_lines_with_chain(pdb_text: str, chain_id: str, serial_start: int) -> tuple[list[str], int]:
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


def overlay_text(entries: list[tuple[str, str, Path]]) -> str:
    lines = [
        "REMARK raw denoise overlay; coordinates are not Kabsch aligned",
    ]
    lines.extend(f"REMARK chain {chain_id} {label}" for chain_id, label, _ in entries)

    serial = 1
    for chain_id, _, pdb_path in entries:
        atom_lines, serial = atom_lines_with_chain(pdb_path.read_text(), chain_id, serial)
        lines.extend(atom_lines)
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_overlay_file(path: Path, entries: list[tuple[str, str, Path]], overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.write_text(overlay_text(entries))
    return True


def write_overlays(sigma_dir: Path, overwrite: bool) -> int:
    written = 0

    target_path = sigma_dir / "target_augmented.pdb"
    pred_path = sigma_dir / "prediction.pdb"
    noisy_path = sigma_dir / "noisy_input.pdb"
    if not target_path.exists() or not pred_path.exists():
        return written

    sigma_name = sigma_dir.name
    domain_dir = sigma_dir.parent
    clean_entries = [
        ("A", "target_augmented", target_path),
        ("B", "prediction", pred_path),
    ]
    if write_overlay_file(domain_dir / f"{sigma_name}_overlay.pdb", clean_entries, overwrite):
        written += 1

    if noisy_path.exists():
        flat_noisy_path = domain_dir / f"{sigma_name}_noisy.pdb"
        if overwrite or not flat_noisy_path.exists():
            flat_noisy_path.write_text(noisy_path.read_text())
            written += 1

    return written


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    probe_dir = run_dir / "denoise_probe"
    if not probe_dir.exists():
        raise FileNotFoundError(f"{probe_dir} does not exist")

    written = 0
    seen = 0
    for sigma_dir in sorted(probe_dir.glob("step_*/*/sigma_*")):
        if not sigma_dir.is_dir():
            continue
        seen += 1
        written += write_overlays(sigma_dir, args.overwrite)

    print(f"run_dir={run_dir}")
    print(f"sigma_dirs_seen={seen}")
    print(f"overlays_written={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
