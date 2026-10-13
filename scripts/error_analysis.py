"""Error analysis on the test split: which signs are missed, confused, or falsely detected.

For every test image, predictions are matched to ground truth (IoU >= 0.5):
  correct       : matched, same class
  wrong_class   : matched box, different class   (e.g. 50 km/h predicted as 60 km/h)
  missed        : ground-truth sign with no prediction
  false_positive: prediction with no ground-truth sign
Writes reports/error_analysis/: summary.md, per_class.csv, and example images per error type
(green = ground truth, colored = prediction).

Example:
    python scripts/error_analysis.py --weights runs/s_960/weights/best.pt
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnsign.classes import PROJECT_ROOT, load_classes  # noqa: E402
from vnsign.inference import SignDetector, draw_detections  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORIES = ("missed", "wrong_class", "false_positive")


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_file: Path, w: int, h: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    gt = []
    for line in label_file.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        xc, yc, bw, bh = (float(v) for v in p[1:5])
        gt.append((int(p[0]), (int((xc - bw / 2) * w), int((yc - bh / 2) * h),
                               int((xc + bw / 2) * w), int((yc + bh / 2) * h))))
    return gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-examples", type=int, default=40, help="example images saved per error type")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "error_analysis")
    args = ap.parse_args()

    classes = load_classes()
    key = {c.id: c.key for c in classes}
    det = SignDetector(args.weights, conf=args.conf, imgsz=args.imgsz)
    pred_id = {c.key: c.id for c in classes}  # model class name (key) -> our id

    for cat in CATEGORIES:
        (args.out / cat).mkdir(parents=True, exist_ok=True)
    saved = Counter()
    stats = defaultdict(Counter)  # per class: tp / fn / fp / wrong
    confusions = Counter()
    missed_sizes = []

    images = sorted(p for p in (args.data / "images" / args.split).iterdir() if p.suffix.lower() in IMG_EXTS)
    for n, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lbl = args.data / "labels" / args.split / f"{img_path.stem}.txt"
        gt = load_gt(lbl, w, h) if lbl.exists() else []
        preds = det.predict(img)

        used = set()
        errors = set()
        for g_cls, g_box in gt:
            best_j, best_iou = None, args.iou
            for j, p in enumerate(preds):
                if j not in used and (v := iou(g_box, p.xyxy)) >= best_iou:
                    best_j, best_iou = j, v
            if best_j is None:
                stats[g_cls]["fn"] += 1
                missed_sizes.append(max(g_box[2] - g_box[0], g_box[3] - g_box[1]))
                errors.add("missed")
                continue
            used.add(best_j)
            p_cls = pred_id.get(preds[best_j].cls.key, -1)
            if p_cls == g_cls:
                stats[g_cls]["tp"] += 1
            else:
                stats[g_cls]["wrong"] += 1
                confusions[(key[g_cls], preds[best_j].cls.key)] += 1
                errors.add("wrong_class")
        for j, p in enumerate(preds):
            if j not in used:
                stats[pred_id.get(p.cls.key, -1)]["fp"] += 1
                errors.add("false_positive")

        for cat in errors:
            if saved[cat] < args.max_examples:
                vis = draw_detections(img, preds)
                for g_cls, (x1, y1, x2, y2) in gt:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"GT:{key[g_cls]}", (x1, min(h - 4, y2 + 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(str(args.out / cat / img_path.name), vis)
                saved[cat] += 1
        if n % 200 == 0:
            print(f"  {n}/{len(images)} images")

    rows = []
    for c in classes:
        s = stats[c.id]
        n_gt = s["tp"] + s["fn"] + s["wrong"]
        if n_gt == 0 and s["fp"] == 0:
            continue
        rows.append({"key": c.key, "name": c.name, "gt": n_gt, "correct": s["tp"], "wrong_class": s["wrong"],
                     "missed": s["fn"], "false_pos": s["fp"],
                     "recall": round(s["tp"] / n_gt, 3) if n_gt else None})
    rows.sort(key=lambda r: (r["recall"] is None, r["recall"] if r["recall"] is not None else 0))
    with open(args.out / "per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    tot = Counter()
    for s in stats.values():
        tot.update(s)
    small = sum(v < 20 for v in missed_sizes)
    md = [
        f"# Phân tích lỗi trên tập {args.split} ({len(images)} ảnh, conf={args.conf}, imgsz={args.imgsz})",
        "",
        f"- Đúng: **{tot['tp']}**, nhầm lớp: **{tot['wrong']}**, bỏ sót: **{tot['fn']}**, báo nhầm: **{tot['fp']}**",
        f"- Trong số biển bị bỏ sót, {small}/{len(missed_sizes)} có cạnh < 20 px (biển rất nhỏ / ở xa)",
        "",
        "## Các cặp hay bị nhầm (thật → dự đoán)",
        "",
        "| Thật | Dự đoán | Số lần |",
        "|---|---|---|",
        *[f"| {a} | {b} | {n} |" for (a, b), n in confusions.most_common(15)],
        "",
        "## 10 lớp có recall thấp nhất",
        "",
        "| Lớp | Tên | Số GT | Recall |",
        "|---|---|---|---|",
        *[f"| {r['key']} | {r['name']} | {r['gt']} | {r['recall']} |" for r in rows if r["recall"] is not None][:10],
        "",
        f"Ảnh ví dụ: {', '.join(f'{c}/ ({saved[c]})' for c in CATEGORIES)}",
    ]
    (args.out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
