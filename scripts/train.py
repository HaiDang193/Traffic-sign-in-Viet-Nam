"""Train a YOLO model with configs/train.yaml, then evaluate on the test split.

Examples:
    python scripts/train.py --model yolo11n.pt --name baseline_n_640 --imgsz 640
    python scripts/train.py --model yolo11s.pt --name s_960
    python scripts/train.py --model yolo11m.pt --name m_1280 --imgsz 1280 --batch 8
Results go to runs/<name>/ (best weights: runs/<name>/weights/best.pt).
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnsign.classes import PROJECT_ROOT  # noqa: E402


def parse_value(v: str):
    return yaml.safe_load(v)  # "0.5" -> 0.5, "true" -> True, "s" -> "s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained weights, or last.pt to resume")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "dataset" / "data.yaml")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train.yaml")
    ap.add_argument("--name", required=True, help="experiment name, e.g. s_960_noflip")
    ap.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs")
    ap.add_argument("--resume", action="store_true")
    args, extra = ap.parse_known_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.resume:
        model.train(resume=True)
    else:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        # Remaining --key value pairs override the yaml config.
        it = iter(extra)
        for k in it:
            if not k.startswith("--"):
                sys.exit(f"Unexpected argument: {k}")
            cfg[k[2:].replace("-", "_")] = parse_value(next(it))
        model.train(data=str(args.data), project=str(args.project), name=args.name, exist_ok=False, **cfg)

    # Evaluate the best checkpoint on the test split (settings read back from the run's args.yaml).
    save_dir = Path(model.trainer.save_dir)
    run_args = yaml.safe_load((save_dir / "args.yaml").read_text(encoding="utf-8"))
    best = YOLO(str(save_dir / "weights" / "best.pt"))
    m = best.val(data=run_args["data"], split="test", imgsz=run_args["imgsz"],
                 project=str(save_dir.parent), name=f"{save_dir.name}_test")
    summary = {
        "name": save_dir.name,
        "model": Path(str(run_args["model"])).name,
        "imgsz": run_args["imgsz"],
        "test_mAP50": round(float(m.box.map50), 4),
        "test_mAP50_95": round(float(m.box.map), 4),
        "test_precision": round(float(m.box.mp), 4),
        "test_recall": round(float(m.box.mr), 4),
        "speed_ms_per_img": {k: round(v, 2) for k, v in m.speed.items()},
    }
    out = save_dir / "test_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()