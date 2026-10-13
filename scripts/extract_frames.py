"""Extract frames from road videos (phone / dashcam) for labeling.

Frames are named <video_stem>_f<frame_idx>.jpg so that split_dataset.py can keep all
frames of one video in the same split (avoids train/test leakage).

Example:
    python scripts/extract_frames.py --videos data/raw_videos --out data/frames --every 0.5
"""
import argparse
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def extract(video: Path, out_dir: Path, every_sec: float, max_side: int) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[skip] cannot open {video}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_sec))
    saved, idx = 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{video.stem}_f{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        idx += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, type=Path, help="video file or folder of videos")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--every", type=float, default=0.5, help="seconds between saved frames (default 0.5)")
    ap.add_argument("--max-side", type=int, default=1920, help="downscale so the longest side <= this")
    args = ap.parse_args()

    videos = [args.videos] if args.videos.is_file() else sorted(
        p for p in args.videos.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for v in videos:
        n = extract(v, args.out, args.every, args.max_side)
        print(f"{v.name}: {n} frames")
        total += n
    print(f"Done: {total} frames from {len(videos)} videos -> {args.out}")


if __name__ == "__main__":
    main()
