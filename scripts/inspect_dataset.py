from __future__ import annotations

import argparse

from drpo.datasets import load_d4rl_hdf5


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a legacy D4RL HDF5 dataset.")
    parser.add_argument("path")
    parser.add_argument("--max-transitions", type=int, default=None)
    args = parser.parse_args()

    batch = load_d4rl_hdf5(args.path, args.max_transitions)
    print(f"transitions: {batch.size}")
    print(f"observations: {batch.observations.shape} {batch.observations.dtype}")
    print(f"actions: {batch.actions.shape} {batch.actions.dtype}")
    print(f"rewards: {batch.rewards.shape} {batch.rewards.dtype}")
    print(f"terminals: {batch.terminals.shape} {batch.terminals.dtype}")
    if batch.next_observations is not None:
        print(f"next_observations: {batch.next_observations.shape} {batch.next_observations.dtype}")
    if batch.timeouts is not None:
        print(f"timeouts: {batch.timeouts.shape} {batch.timeouts.dtype}")


if __name__ == "__main__":
    main()
