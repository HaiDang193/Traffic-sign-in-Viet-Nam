"""Run the trained model on images or a video.

Image / folder  -> outputs/<name>/<image>.jpg (annotated) + detections.json
Video           -> outputs/<name>/<video>_annotated.mp4 + signs_log.csv (one row per sign) + crops/

Examples:
    python scripts/predict.py --weights runs/s_960/weights/best.pt --source samples/street.jpg
    python scripts/predict.py --weights runs/s_960/weights/best.pt --source data/test_videos/route1.mp4
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnsign.classes import PROJECT_ROOT  # noqa: E402
from vnsign.inference import SignDetector, SignLogger, draw_detections  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def run_images(det: SignDetector, paths: list[Path], out: Path):
    results = {}
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[skip] cannot read {p}")
            continue
        dets = det.predict(img)
        cv2.imwrite(str(out / p.name), draw_detections(img, dets))
        results[p.name] = [d.to_dict() for d in dets]
        print(f"{p.name}: " + (", ".join(f"{d.cls.code} {d.conf:.2f}" for d in dets) or "no sign"))
    (out / "detections.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def run_video(det: SignDetector, path: Path, out: Path, min_hits: int):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out / f"{path.stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    logger = SignLogger(fps, min_hits=min_hits)
    det.reset_tracker()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.track(frame)
        logger.update(idx, frame, dets)
        writer.write(draw_detections(frame, dets))
        idx += 1
        if idx % 100 == 0:
            print(f"  frame {idx}/{n_frames}")
    cap.release()
    writer.release()

    rows = logger.rows()
    crops = out / "crops"
    crops.mkdir(exist_ok=True)
    for r in rows:
        crop = logger.crop(r["track_id"])
        if crop is not None and crop.size:
            cv2.imwrite(str(crops / f"{r['time_sec']:08.2f}_{r['key']}_id{r['track_id']}.jpg"), crop)
    with open(out / "signs_log.csv", "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: opens in Excel
        fields = ["time", "time_sec", "track_id", "code", "key", "name", "meaning", "confidence", "frames_seen"]
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} signs logged -> {out / 'signs_log.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path, help="image, folder of images, or video")
    ap.add_argument("--out", type=Path, help="output folder (default outputs/<source name>)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--min-hits", type=int, default=3, help="video: frames a sign must be tracked to be logged")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu")
    args = ap.parse_args()

    out = args.out or PROJECT_ROOT / "outputs" / args.source.stem
    out.mkdir(parents=True, exist_ok=True)
    det = SignDetector(args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device)

    if args.source.is_dir():
        run_images(det, sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMG_EXTS), out)
    elif args.source.suffix.lower() in VIDEO_EXTS:
        run_video(det, args.source, out, args.min_hits)
    else:
        run_images(det, [args.source], out)
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
