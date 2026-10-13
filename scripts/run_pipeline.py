"""Run the whole pipeline with one command:

    download datasets -> split -> dataset stats -> train (+ test eval) -> error analysis
    -> comparison table -> copy best.pt to models/ for the app

Every step can be skipped, so it is also handy for re-training with other settings.

Examples:
    python scripts/run_pipeline.py                                   # defaults: yolo11s, imgsz 960, 100 epochs
    python scripts/run_pipeline.py --model yolo11n.pt --imgsz 640 --epochs 30 --name n_640_quick
    python scripts/run_pipeline.py --skip-data --model yolo11m.pt --name m_960 --batch 8
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(title: str, cmd: list[str]):
    print(f"\n{'=' * 70}\n>>> {title}\n    {' '.join(cmd)}\n{'=' * 70}", flush=True)
    if subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode != 0:
        sys.exit(f"Step failed: {title}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--name", help="run name (default: <model>_<imgsz>)")
    ap.add_argument("--skip-data", action="store_true", help="dataset/ already built: skip download + split")
    ap.add_argument("--skip-train", action="store_true", help="only (re)build the data")
    args = ap.parse_args()
    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"

    if not args.skip_data:
        step("1. Tải + import dataset", ["scripts/download_datasets.py"])
        step("2. Chia train/val/test", ["scripts/split_dataset.py", "--force"])
        step("3. Thống kê dataset", ["scripts/dataset_stats.py"])
    if args.skip_train:
        return

    if (ROOT / "runs" / name).exists():
        sys.exit(f"runs/{name} already exists: choose another --name (or delete that folder)")
    step("4. Train + đánh giá trên test", ["scripts/train.py", "--model", args.model, "--name", name,
                                          "--imgsz", str(args.imgsz), "--epochs", str(args.epochs),
                                          "--batch", str(args.batch)])
    best = ROOT / "runs" / name / "weights" / "best.pt"
    step("5. Phân tích lỗi", ["scripts/error_analysis.py", "--weights", str(best), "--imgsz", str(args.imgsz),
                             "--out", str(ROOT / "reports" / "error_analysis" / name)])
    step("6. Bảng so sánh các lần train", ["scripts/compare_runs.py"])

    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models" / f"{name}.pt")
    print(f"\nDone. Model copied to models/{name}.pt -> run the demo: streamlit run app/app.py")


if __name__ == "__main__":
    main()
