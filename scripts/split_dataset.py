"""Build the training dataset (dataset/) from everything in data/pool/ and write dataset/data.yaml.

Images are split by GROUP, not individually: all frames from one video (<video>_f000123)
and all Roboflow augmentations of one photo (<photo>_jpg.rf.<hash>) go to the same split.
Otherwise near-identical frames leak into val/test and the metrics look much better than reality.

Example:
    python scripts/split_dataset.py --ratios 0.75 0.15 0.10 --force
"""
import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnsign.classes import PROJECT_ROOT, load_classes  # noqa: E402

SPLITS = ("train", "val", "test")
GROUP_SUFFIX = re.compile(r"(_f\d+|_(jpg|jpeg|png)\.rf\.[0-9a-zA-Z]+|\.rf\.[0-9a-zA-Z]+)$")


def group_key(stem: str) -> str:
    return GROUP_SUFFIX.sub("", stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=PROJECT_ROOT / "data" / "pool")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "dataset")
    ap.add_argument("--sources", nargs="*", help="only use these pool sub-folders (default: all)")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.75, 0.15, 0.10], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="hard-link instead of copy (same drive only)")
    ap.add_argument("--force", action="store_true", help="delete an existing output folder first")
    args = ap.parse_args()

    if args.out.exists():
        if not args.force:
            sys.exit(f"{args.out} exists. Use --force to rebuild it.")
        shutil.rmtree(args.out)

    sources = [p for p in sorted(args.pool.iterdir()) if p.is_dir()] if args.pool.exists() else []
    if args.sources:
        sources = [p for p in sources if p.name in args.sources]
    if not sources:
        sys.exit(f"No sources in {args.pool}. Run import_dataset.py first.")

    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for src in sources:
        for img in sorted((src / "images").iterdir()):
            lbl = src / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                groups[f"{src.name}/{group_key(img.stem)}"].append((img, lbl))

    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    total = sum(len(v) for v in groups.values())
    r_train, r_val, _ = (r / sum(args.ratios) for r in args.ratios)
    assigned = {s: [] for s in SPLITS}
    done = 0
    for k in keys:
        frac = done / total
        split = "train" if frac < r_train else "val" if frac < r_train + r_val else "test"
        assigned[split].extend(groups[k])
        done += len(groups[k])

    put = (lambda a, b: b.hardlink_to(a)) if args.link else shutil.copy2
    for split, items in assigned.items():
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
        for img, lbl in items:
            put(img, args.out / "images" / split / img.name)
            put(lbl, args.out / "labels" / split / lbl.name)

    classes = load_classes()
    # No `path:` key -> Ultralytics resolves the folders relative to this yaml file,
    # so the same dataset folder works on Windows and on Colab.
    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {c.id: c.key for c in classes},
    }
    with open(args.out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"{len(keys)} groups, {total} images from {[s.name for s in sources]}")
    for split in SPLITS:
        print(f"  {split:<5} {len(assigned[split])} images")
    print(f"Wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
