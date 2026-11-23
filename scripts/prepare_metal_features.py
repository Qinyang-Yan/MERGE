#!/usr/bin/env python
"""Validate and optionally aggregate an electrochemical descriptor table."""

import argparse
from pathlib import Path

import pandas as pd

from merge.data.metal_features import ALL_COLS, prepare_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV containing replicate measurements")
    parser.add_argument("--output", required=True, help="Destination CSV; must differ from input")
    parser.add_argument("--mode", choices=["metal_class", "sample_id"], default="metal_class")
    args = parser.parse_args()
    source = Path(args.input).expanduser()
    destination = Path(args.output).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("Refusing to overwrite the source descriptor table")
    table = pd.read_csv(source)
    prepared = prepare_feature_table(table, mode=args.mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(destination, index=False)
    print(f"Wrote {len(prepared)} validated descriptor rows to {destination}")
    print(prepared.groupby("metal_class")["sample_id"].count().to_string())
    print(prepared[ALL_COLS].agg(["min", "max", "mean"]).to_string())


if __name__ == "__main__":
    main()
