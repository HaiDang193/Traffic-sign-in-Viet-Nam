"""Import an external YOLO-format dataset (e.g. a Roboflow export) into data/pool/<name>/,
remapping its class names to the keys in configs/classes.yaml.

The source must contain a data.yaml (with `names`) and folders .../labels/*.txt next to
.../images/*. Splits of the source (train/valid/test) are merged; split_dataset.py re-splits.

Mapping file (yaml): source class name -> our key, or null to drop those boxes:
    "Cam nguoc chieu": p102
    "Toc do toi da 50": p127
    "car": null
Without --mapping, source names must already equal our keys (e.g. your own Roboflow project).

Example:
    python scripts/import_dataset.py --src downloads/vn_signs_rf --name rf_vnsign \
        --mapping configs/mappings/rf_vnsign.yaml --unmapped other
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnsign.classes import PROJECT_ROOT, key_to_id, load_classes  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_source_names(src: Path) -> dict[int, str]:
    yamls = list(src.glob("data.yaml")) or list(src.rglob("data.yaml"))
    if not yamls:
        sys.exit(f"No data.yaml found in {src}")
    names = yaml.safe_load(yamls[0].read_text(encoding="utf-8"))["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def find_image(label_file: Path) -> Path | None:
    img_dir = label_file.parent.parent / "images"
    for ext in IMG_EXTS:
        for cand in (img_dir / (label_file.stem + ext), img_dir / (label_file.stem + ext.upper())):
            if cand.exists():
                return cand
    return None


def to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """YOLO bbox (4 values) or polygon (x1 y1 x2 y2 ...) -> (xc, yc, w, h)."""
    if len(values) == 4:
        return tuple(values)
    xs, ys = values[0::2], values[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--name", required=True, help="short source name, used as folder + filename prefix")
    ap.add_argument("--mapping", type=Path)
    ap.add_argument("--unmapped", choices=["error", "other", "drop"], default="error",
                    help="what to do with source classes missing from the mapping")
    ap.add_argument("--drop-empty", action="store_true", help="skip images left with no boxes")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "pool")
    args = ap.parse_args()

    ours = key_to_id(load_classes())
    src_names = load_source_names(args.src)
    mapping = (yaml.safe_load(args.mapping.read_text(encoding="utf-8")) or {}) if args.mapping \
        else {n: n for n in src_names.values() if n in ours}

    src_to_ours: dict[int, int | None] = {}
    problems = []
    for sid, sname in src_names.items():
        if sname in mapping:
            target = mapping[sname]
            if target is not None and target not in ours:
                problems.append(f"'{sname}' -> '{target}' (not a key in classes.yaml)")
            src_to_ours[sid] = None if target is None else ours.get(target)
        elif args.unmapped == "other":
            src_to_ours[sid] = ours["other"]
        elif args.unmapped == "drop":
            src_to_ours[sid] = None
        else:
            problems.append(f"'{sname}' has no mapping (add it to --mapping, or use --unmapped other/drop)")
    if problems:
        sys.exit("Fix the mapping first:\n  " + "\n  ".join(problems))

    if (args.out / args.name).exists():  # derived data: re-importing replaces it
        shutil.rmtree(args.out / args.name)
    out_img = args.out / args.name / "images"
    out_lbl = args.out / args.name / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    counts, n_img, n_skipped, n_missing = Counter(), 0, 0, 0
    for lf in sorted(args.src.rglob("labels/*.txt")):
        img = find_image(lf)
        if img is None:
            n_missing += 1
            continue
        lines = []
        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            new_id = src_to_ours.get(int(float(parts[0])))
            if new_id is None:
                continue
            xc, yc, w, h = to_bbox([float(v) for v in parts[1:]])
            lines.append(f"{new_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            counts[new_id] += 1
        if not lines and args.drop_empty:
            n_skipped += 1
            continue
        stem = f"{args.name}__{lf.stem}"
        shutil.copy2(img, out_img / (stem + img.suffix.lower()))
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1

    id_to_key = {v: k for k, v in ours.items()}
    print(f"Imported {n_img} images into {args.out / args.name} "
          f"(skipped empty: {n_skipped}, labels without image: {n_missing})")
    for cid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {id_to_key[cid]:<8} {n}")


if __name__ == "__main__":
    main()
