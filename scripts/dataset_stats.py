"""EDA for dataset/: boxes per class per split, box size distribution, empty images.
Saves charts to reports/figures/ for the report.

Example:
    python scripts/dataset_stats.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnsign.classes import PROJECT_ROOT, load_classes  # noqa: E402

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "dataset")
    ap.add_argument("--figs", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    ap.add_argument("--min-boxes", type=int, default=100, help="warn for classes with fewer train boxes")
    args = ap.parse_args()

    classes = load_classes()
    counts = {s: Counter() for s in SPLITS}
    sizes = []  # sqrt(w*h) relative to image, i.e. box side as a fraction of the image
    empty = Counter()
    n_img = Counter()
    for split in SPLITS:
        for lf in (args.data / "labels" / split).glob("*.txt"):
            n_img[split] += 1
            lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
            if not lines:
                empty[split] += 1
            for p in lines:
                counts[split][int(p[0])] += 1
                sizes.append((float(p[3]) * float(p[4])) ** 0.5)

    print(f"{'class':<8} {'code':<7} {'train':>6} {'val':>5} {'test':>5}  name")
    for c in classes:
        tr, va, te = (counts[s][c.id] for s in SPLITS)
        flag = "  <-- ít dữ liệu" if tr < args.min_boxes else ""
        print(f"{c.key:<8} {c.code:<7} {tr:>6} {va:>5} {te:>5}  {c.name}{flag}")
    for s in SPLITS:
        print(f"{s}: {n_img[s]} images, {sum(counts[s].values())} boxes, {empty[s]} without boxes")
    if sizes:
        small = sum(v < 0.03 for v in sizes) / len(sizes)
        hint = " -> consider imgsz >= 960 / SAHI" if small > 0.2 else ""
        print(f"Box side < 3% of image (very small): {small:.0%}{hint}")

    args.figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([c.key for c in classes], [counts["train"][c.id] for c in classes])
    ax.axhline(args.min_boxes, color="red", linestyle="--", linewidth=1)
    ax.set_title("Số bbox mỗi lớp (train)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(args.figs / "class_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sizes, bins=50)
    ax.set_xlabel("Cạnh bbox / cạnh ảnh")
    ax.set_title("Phân bố kích thước biển báo")
    fig.tight_layout()
    fig.savefig(args.figs / "box_sizes.png", dpi=150)
    print(f"Charts saved to {args.figs}")


if __name__ == "__main__":
    main()
