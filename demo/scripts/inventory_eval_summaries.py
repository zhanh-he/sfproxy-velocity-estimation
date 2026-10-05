"""Index small evaluation summaries from a research workspace into CSV.

This inventories saved runs. It does not select the paper's final runs.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"real_vs_synth_pred_(bssl|bstl)\n\s*pearson[^\n]*\n\s*([0-9.]+)"
)


def extract(path: Path, workspace: Path):
    content = path.read_text(errors="replace")
    run = path.parent.relative_to(workspace)
    result = {
        "run": str(run),
        "route": run.parts[0],
        "dataset": run.parts[1],
        "variant": run.parts[2],
        "checkpoint": "",
        "num_items": "",
        "num_ok": "",
        "num_fail": "",
        "velocity_mae": "",
        "r_bssl": "",
        "r_bstl": "",
    }
    match = re.search(r"CHECKPOINT_PATH = (.+)", content)
    if match:
        checkpoint = Path(match.group(1).strip())
        try:
            result["checkpoint"] = str(checkpoint.relative_to(workspace))
        except ValueError:
            result["checkpoint"] = checkpoint.name
    match = re.search(r"num_items/ok/fail:\s*(\d+)/(\d+)/(\d+)", content)
    if match:
        result["num_items"], result["num_ok"], result["num_fail"] = match.groups()
    match = re.search(r"velocity_mae \(0-127\):\s*([0-9.]+)", content)
    if match:
        result["velocity_mae"] = match.group(1)
    for kind, value in METRIC_RE.findall(content):
        result[f"r_{kind}"] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    records = [extract(path, workspace) for path in sorted(workspace.glob("route*_eval/*/*/result_summary.txt"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else ["run"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Indexed {len(records)} evaluation summaries")


if __name__ == "__main__":
    main()
