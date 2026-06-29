#!/usr/bin/env python3
"""Transfer mdCATH domains from storizzle filtered to length <= 200 and 320 K.

The source files on storizzle contain all temperatures inside each HDF5 file.
This script runs a small h5py filter on storizzle first, stages HDF5 files that
contain only eligible domains and only temperature group 320, then rsyncs the
staged result into this checkout's data directory.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_ROOT = REPO_ROOT / "data" / "mdcath_320K_len_le200"
DEFAULT_REMOTE_ROOT = "/volume1/homes/theodor/MDcath_download/mdcath_minimal_h5"
DEFAULT_REMOTE_STAGE_ROOT = "/volume1/homes/theodor/MDcath_download/mdcath_transfer_staging"


REMOTE_FILTER_SCRIPT = r'''
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import traceback

try:
    import h5py
except ImportError as err:
    print(
        "remote error: this script needs h5py installed on storizzle "
        "(python3 -m pip install h5py)",
        file=sys.stderr,
    )
    raise SystemExit(2) from err


def as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if hasattr(value, "shape") and value.shape == ():
        return as_text(value.item())
    if hasattr(value, "tolist"):
        return as_text(value.tolist())
    if isinstance(value, (list, tuple)):
        return "".join(as_text(item) for item in value)
    return str(value)


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def source_files(data_dir):
    return sorted(data_dir.glob("mdcath_dataset_*.h5")) + sorted(
        data_dir.glob("mdcath_dataset_*.hd5")
    )


def copy_filtered_domain(src_h5, domain_name, temperature, out_path):
    domain = src_h5[domain_name]
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(tmp_path, "w") as dst_h5:
        copy_attrs(src_h5, dst_h5)
        dst_domain = dst_h5.create_group(domain_name)
        copy_attrs(domain, dst_domain)

        for key in domain.keys():
            if key == temperature or not key.isdigit():
                domain.copy(key, dst_domain)

        temp_groups = sorted(key for key in dst_domain.keys() if key.isdigit())
        if temp_groups != [temperature]:
            raise ValueError(
                f"expected only temperature group {temperature}, found {temp_groups}"
            )

    tmp_path.replace(out_path)


def main():
    source_root = Path(sys.argv[1])
    stage_root = Path(sys.argv[2])
    temperature = sys.argv[3]
    max_length_inclusive = int(sys.argv[4])
    dry_run = sys.argv[5] == "1"

    data_dir = source_root / "data"
    out_data_dir = stage_root / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    files = source_files(data_dir)

    for source_path in files:
        try:
            with h5py.File(source_path, "r") as src_h5:
                domain_names = sorted(
                    name
                    for name, item in src_h5.items()
                    if isinstance(item, h5py.Group)
                )
                if not domain_names:
                    entries.append(
                        {
                            "source": str(source_path),
                            "domain": "",
                            "status": "skipped",
                            "reason": "no_domain_groups",
                        }
                    )
                    continue

                for domain_name in domain_names:
                    entry = {
                        "source": str(source_path),
                        "domain": domain_name,
                        "temperature": temperature,
                        "max_length_inclusive": max_length_inclusive,
                    }

                    domain = src_h5[domain_name]
                    if "sequence" not in domain:
                        entry.update(status="skipped", reason="missing_sequence")
                        entries.append(entry)
                        continue

                    sequence = as_text(domain["sequence"][()])
                    sequence_length = len(sequence)
                    entry["sequence_length"] = sequence_length

                    if sequence_length > max_length_inclusive:
                        entry.update(status="skipped", reason="sequence_too_long")
                        entries.append(entry)
                        continue

                    if temperature not in domain:
                        entry.update(status="skipped", reason="missing_temperature")
                        entries.append(entry)
                        continue

                    out_path = out_data_dir / f"mdcath_dataset_{domain_name}.h5"
                    entry["output"] = str(out_path)

                    if dry_run:
                        entry.update(status="would_copy", reason="")
                    else:
                        copy_filtered_domain(src_h5, domain_name, temperature, out_path)
                        entry.update(status="copied", reason="")

                    entries.append(entry)
        except Exception as err:
            entries.append(
                {
                    "source": str(source_path),
                    "domain": "",
                    "status": "error",
                    "reason": f"{type(err).__name__}: {err}",
                    "traceback": traceback.format_exc(),
                }
            )

    status_counts = Counter(entry["status"] for entry in entries)
    selected_status = "would_copy" if dry_run else "copied"
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "stage_root": str(stage_root),
        "temperature": temperature,
        "max_length_inclusive": max_length_inclusive,
        "dry_run": dry_run,
        "source_file_count": len(files),
        "domain_count": len(entries),
        "selected_domain_count": status_counts[selected_status],
        "status_counts": dict(sorted(status_counts.items())),
        "entries": entries,
    }

    manifest_path = stage_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    tsv_path = stage_root / "manifest.tsv"
    columns = [
        "status",
        "reason",
        "domain",
        "sequence_length",
        "temperature",
        "source",
        "output",
    ]
    with tsv_path.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for entry in entries:
            handle.write(
                "\t".join(str(entry.get(column, "")) for column in columns) + "\n"
            )

    printable = dict(summary)
    printable.pop("entries")
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def remote_target(user: str, host: str) -> str:
    return f"{user}@{host}" if user else host


def quoted_remote_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run_command(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    capture_output: bool = False,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("+", shlex.join(command), file=sys.stderr)

    return subprocess.run(
        list(command),
        input=input_text,
        text=True,
        capture_output=capture_output,
        check=True,
    )


def ssh_command(args: argparse.Namespace) -> list[str]:
    command = ["ssh", "-o", "BatchMode=yes"]

    identity = args.identity.expanduser() if args.identity else None
    if identity and identity.exists():
        command.extend(["-i", str(identity), "-o", "IdentitiesOnly=yes"])

    if args.known_hosts:
        command.extend(
            [
                "-o",
                f"UserKnownHostsFile={args.known_hosts.expanduser()}",
                "-o",
                "StrictHostKeyChecking=yes",
            ]
        )

    for option in args.ssh_option:
        command.extend(["-o", option])

    return command


def create_remote_stage(args: argparse.Namespace, ssh: Sequence[str], target: str) -> str:
    stage_root = args.remote_stage_root.rstrip("/")
    remote_command = (
        f"mkdir -p {shlex.quote(stage_root)} && "
        f"mktemp -d {shlex.quote(stage_root)}/mdcath_320k_le200.XXXXXXXXXX"
    )
    result = run_command(
        [*ssh, target, remote_command],
        capture_output=True,
        verbose=args.verbose,
    )
    stage_path = result.stdout.strip()
    if not stage_path:
        raise RuntimeError("storizzle did not return a remote staging path")
    expected_prefix = f"{stage_root}/mdcath_320k_le200."
    if not stage_path.startswith(expected_prefix):
        raise RuntimeError(f"unexpected remote staging path: {stage_path}")
    return stage_path


def cleanup_remote_stage(
    args: argparse.Namespace, ssh: Sequence[str], target: str, stage_path: str
) -> None:
    stage_root = args.remote_stage_root.rstrip("/")
    expected_prefix = f"{stage_root}/mdcath_320k_le200."
    if not stage_path.startswith(expected_prefix):
        print(f"not cleaning unexpected remote staging path: {stage_path}", file=sys.stderr)
        return

    remote_command = (
        f"case {shlex.quote(stage_path)} in "
        f"{shlex.quote(expected_prefix)}*) rm -rf -- {shlex.quote(stage_path)} ;; "
        "*) echo 'refusing to clean unexpected path' >&2; exit 2 ;; "
        "esac"
    )
    run_command([*ssh, target, remote_command], verbose=args.verbose)


def filter_on_remote(
    args: argparse.Namespace, ssh: Sequence[str], target: str, stage_path: str
) -> None:
    remote_args = [
        args.remote_python,
        "-",
        args.source_root,
        stage_path,
        str(args.temperature),
        str(args.max_length),
        "1" if args.dry_run else "0",
    ]
    result = run_command(
        [*ssh, target, quoted_remote_command(remote_args)],
        input_text=REMOTE_FILTER_SCRIPT,
        capture_output=True,
        verbose=args.verbose,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")


def rsync_stage(args: argparse.Namespace, ssh: Sequence[str], target: str, stage_path: str) -> None:
    local_root = args.local_root.expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    ssh_transport = quoted_remote_command(ssh)
    source = f"{target}:{stage_path}/"
    destination = f"{local_root}/"

    command = [
        "rsync",
        "-av",
        "--prune-empty-dirs",
        f"--rsync-path={args.rsync_path}",
        "-e",
        ssh_transport,
        "--exclude=*.tmp",
        source,
        destination,
    ]
    run_command(command, verbose=args.verbose)


def rsync_manifests(
    args: argparse.Namespace, ssh: Sequence[str], target: str, stage_path: str
) -> None:
    local_root = args.local_root.expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    ssh_transport = quoted_remote_command(ssh)
    source = f"{target}:{stage_path}/"
    destination = f"{local_root}/"

    command = [
        "rsync",
        "-av",
        f"--rsync-path={args.rsync_path}",
        "-e",
        ssh_transport,
        "--include=/manifest.json",
        "--include=/manifest.tsv",
        "--exclude=*",
        source,
        destination,
    ]
    run_command(command, verbose=args.verbose)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage and transfer mdCATH HDF5 domains from storizzle where "
            "sequence length is <= 200 amino acids, keeping only temperature 320 K."
        )
    )
    parser.add_argument("--host", default="storizzle", help="SSH host for storizzle")
    parser.add_argument("--user", default="theodor", help="SSH user")
    parser.add_argument("--source-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--remote-stage-root", default=DEFAULT_REMOTE_STAGE_ROOT)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--temperature", default="320")
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Keep domains with sequence_length <= this value",
    )
    parser.add_argument(
        "--identity",
        type=Path,
        default=Path.home() / ".ssh" / "id_ed25519",
        help="SSH private key. If the file does not exist, normal SSH auth is used.",
    )
    parser.add_argument(
        "--known-hosts",
        type=Path,
        default=None,
        help="Optional known_hosts file, e.g. /tmp/storizzle_known_hosts",
    )
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra SSH -o option. May be repeated.",
    )
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--rsync-path", default="/bin/rsync")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect remote HDF5 files and copy only manifest files locally.",
    )
    parser.add_argument(
        "--keep-remote-staging",
        action="store_true",
        help="Do not delete the remote filtered staging directory after transfer.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = remote_target(args.user, args.host)
    ssh = ssh_command(args)
    stage_path = ""

    try:
        stage_path = create_remote_stage(args, ssh, target)
        print(f"remote staging: {target}:{stage_path}")

        filter_on_remote(args, ssh, target, stage_path)

        if args.dry_run:
            rsync_manifests(args, ssh, target, stage_path)
            print(f"dry run complete; manifests copied to {args.local_root.expanduser().resolve()}")
            return 0

        rsync_stage(args, ssh, target, stage_path)
        print(f"local output: {args.local_root.expanduser().resolve()}")
        return 0
    except subprocess.CalledProcessError as err:
        print(f"command failed with exit code {err.returncode}: {shlex.join(err.cmd)}", file=sys.stderr)
        if err.stdout:
            print(err.stdout, file=sys.stderr, end="")
        if err.stderr:
            print(err.stderr, file=sys.stderr, end="")
        return err.returncode or 1
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    finally:
        if stage_path and not args.keep_remote_staging:
            try:
                cleanup_remote_stage(args, ssh, target, stage_path)
            except subprocess.CalledProcessError as err:
                print(
                    f"warning: could not clean remote staging path {stage_path}: {err}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
