"""Collect the results of all training runs into one comparison table for the report.

Reads runs/*/test_summary.json (written by train.py) and runs/*/results.csv, writes
reports/experiments.csv and reports/experiments.md (paste the markdown into the report).

Example:
    python scripts/compare_runs.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnsign.classes import PROJECT_ROOT  # noqa: E402

COLUMNS = ["name", "model", "imgsz", "epochs", "fliplr", "best_epoch", "val_mAP50_95",
           "test_mAP50", "test_mAP50_95", "test_precision", "test_recall", "inference_ms"]


def best_val(results_csv: Path) -> tuple[int | None, float | None]:
    if not results_csv.exists():
        return None, None
    with open(results_csv, encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    rows = [r for r in rows if r.get("metrics/mAP50-95(B)")]
    if not rows:
        return None, None
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return int(float(best["epoch"])), round(float(best["metrics/mAP50-95(B)"]), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports")
    args = ap.parse_args()

    rows = []
    for summary_file in sorted(args.runs.glob("*/test_summary.json")):
        run = summary_file.parent
        s = json.loads(summary_file.read_text())
        train_args = yaml.safe_load((run / "args.yaml").read_text()) if (run / "args.yaml").exists() else {}
        best_epoch, val_map = best_val(run / "results.csv")
        rows.append({
            "name": s["name"],
            "model": Path(str(s.get("model", train_args.get("model", "")))).name,
            "imgsz": s.get("imgsz"),
            "epochs": train_args.get("epochs"),
            "fliplr": train_args.get("fliplr"),
            "best_epoch": best_epoch,
            "val_mAP50_95": val_map,
            "test_mAP50": s["test_mAP50"],
            "test_mAP50_95": s["test_mAP50_95"],
            "test_precision": s["test_precision"],
            "test_recall": s["test_recall"],
            "inference_ms": s["speed_ms_per_img"].get("inference"),
        })
    if not rows:
        sys.exit(f"No runs with test_summary.json in {args.runs}")
    rows.sort(key=lambda r: -r["test_mAP50_95"])

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "experiments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    md += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in COLUMNS) + " |" for r in rows]
    (args.out / "experiments.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {args.out / 'experiments.csv'} and experiments.md")


if __name__ == "__main__":
    main()
