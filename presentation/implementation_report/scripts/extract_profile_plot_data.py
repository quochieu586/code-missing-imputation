"""Extract a deterministic plot series from the saved v2 truth arrays.

This reads existing NPZ bundles only. It does not import or call the generator,
fit a model, evaluate errors, or write to the source bundles.
Selection rule: lexicographically first declared entity (entity_001), central
named component (component_06, zero-based index 5), all 120 time points.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BUNDLE_ROOT = ROOT / "datasets" / "synthetic" / "wave_panel_v2"
OUTPUT = ROOT / "presentation" / "implementation_report" / "data" / "profile_entity001_component06.csv"
PROFILES = ("easy", "medium", "hard")
ENTITY_ID = "entity_001"
COMPONENT_INDEX = 5
COMPONENT_ID = "component_06"


def row_for_entity(row_ids: np.ndarray, entity_id: str) -> int:
    matches = [
        index
        for index, encoded in enumerate(row_ids[:, 0])
        if json.loads(str(encoded))[0] == entity_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {entity_id}; found {len(matches)}")
    return matches[0]


def main() -> None:
    series = {}
    shared_times = None
    for profile in PROFILES:
        bundle = BUNDLE_ROOT / profile / "private" / "ground_truth.npz"
        with np.load(bundle, allow_pickle=False) as arrays:
            row = row_for_entity(arrays["row_ids"], ENTITY_ID)
            ids = [json.loads(str(encoded)) for encoded in arrays["row_ids"][row]]
            if any(item[0] != ENTITY_ID for item in ids):
                raise ValueError(f"Mixed entity rows in {bundle}")
            times = [item[1] for item in ids]
            if shared_times is None:
                shared_times = times
            elif times != shared_times:
                raise ValueError(f"Time axis differs in {bundle}")
            totals = arrays["N"][row].astype(np.float64)
            if np.any(totals <= 0):
                raise ValueError(f"Nonpositive total in {bundle}")
            series[profile] = {
                "p": arrays["p"][row, :, COMPONENT_INDEX].astype(np.float64),
                "x_over_n": arrays["counts"][row, :, COMPONENT_INDEX].astype(np.float64) / totals,
                "N": arrays["N"][row].astype(np.int64),
            }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    header = ["time_index"]
    for profile in PROFILES:
        header.extend([f"{profile}_p", f"{profile}_x_over_n", f"{profile}_N"])
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for time_index in range(len(shared_times)):
            row = [time_index]
            for profile in PROFILES:
                values = series[profile]
                row.extend([
                    f"{values['p'][time_index]:.10f}",
                    f"{values['x_over_n'][time_index]:.10f}",
                    int(values["N"][time_index]),
                ])
            writer.writerow(row)
    print(f"Wrote {len(shared_times)} existing-array rows to {OUTPUT.relative_to(ROOT)}")
    print(f"Selection: {ENTITY_ID}, {COMPONENT_ID}; profiles={','.join(PROFILES)}; seed=2026")


if __name__ == "__main__":
    main()