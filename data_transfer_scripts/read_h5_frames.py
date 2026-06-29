#!/usr/bin/env python3
"""Read selected coordinate frames from one mdCATH HDF5 trajectory.

Example:
    python read_h5_frames.py 12asA00 320 1 100 107

Returns an array shaped as:
    (requested_frames, atoms, xyz)
"""

from pathlib import Path
import argparse


def dataset_path(domain, temperature, run):
    return f"{domain}/{temperature}/{run}/coords"


def find_axis(shape, size, name):
    matches = [axis for axis, axis_size in enumerate(shape) if axis_size == size]
    if len(matches) != 1:
        raise ValueError(f"could not identify {name} axis in shape {shape}")
    return matches[0]


def read_frames(domain, temperature, run, frames, data_dir):
    try:
        import h5py
        import numpy as np
    except ImportError as err:
        raise RuntimeError("This script needs h5py and numpy: python -m pip install h5py numpy") from err

    h5_path = data_dir / f"mdcath_dataset_{domain}.h5"

    with h5py.File(h5_path, "r") as h5:
        domain_group = h5[domain]
        run_group = h5[f"{domain}/{temperature}/{run}"]
        coords = h5[dataset_path(domain, temperature, run)]

        num_atoms = int(domain_group.attrs["numProteinAtoms"])
        num_frames = int(run_group.attrs["numFrames"])

        xyz_axis = find_axis(coords.shape, 3, "xyz")
        atom_axis = find_axis(coords.shape, num_atoms, "atom")
        frame_axis = find_axis(coords.shape, num_frames, "frame")

        frames = np.array(frames, dtype=int)
        sorted_frames, restore_order = np.unique(frames, return_inverse=True)

        selection = [slice(None)] * coords.ndim
        selection[frame_axis] = sorted_frames
        data = coords[tuple(selection)]

        data = np.take(data, restore_order, axis=frame_axis)
        data = np.moveaxis(data, (frame_axis, atom_axis, xyz_axis), (0, 1, 2))
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", help="Domain id, e.g. 12asA00")
    parser.add_argument("temperature", help="Temperature, e.g. 320")
    parser.add_argument("run", help="Run/replica id, e.g. 1")
    parser.add_argument("frames", nargs="+", type=int, help="Zero-based frame indices")
    parser.add_argument("--data-dir", type=Path, default=Path("mdcath_minimal_h5/data"))
    args = parser.parse_args()

    data = read_frames(args.domain, args.temperature, args.run, args.frames, args.data_dir)
    print("shape:", data.shape)
    print("first requested frame, first atom xyz:", data[0, 0])


if __name__ == "__main__":
    main()
