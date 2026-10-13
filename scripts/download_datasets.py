"""Download every dataset listed in configs/datasets.yaml, then import it into data/pool/.

Roboflow datasets need a (free) API key: roboflow.com -> Settings -> API Keys.
Put it in a file `.env` at the project root (already git-ignored):
    ROBOFLOW_API_KEY=xxxxxxxx
or set the environment variable ROBOFLOW_API_KEY.

For each dataset:
  1. download into downloads/<name>/ (skipped if already there, unless --force)
  2. print its class names; if the mapping file does not exist yet, write a draft
     (names that equal a QCVN code / key / Vietnamese name are pre-filled)
  3. if the mapping is finished, run import_dataset.py -> data/pool/<name>/

Examples:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --only rf_vn_traffic --force
Then: python scripts/split_dataset.py --force
"""
import argparse
import os
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnsign.classes import PROJECT_ROOT, load_classes  # noqa: E402

DOWNLOADS = PROJECT_ROOT / "downloads"
DRAFT_MARKER = "MAPPING_DRAFT"


def api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    env_file = PROJECT_ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("Missing ROBOFLOW_API_KEY. Create .env with ROBOFLOW_API_KEY=... (see .env.example).")
    return key


def download_roboflow(ds: dict, dest: Path):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key())
    version = rf.workspace(ds["workspace"]).project(ds["project"]).version(int(ds["version"]))
    version.download("yolov8", location=str(dest), overwrite=True)


def download_url(ds: dict, dest: Path):
    zip_path = DOWNLOADS / "_zips" / f"{ds['name']}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {ds['url']}")
    urllib.request.urlretrieve(ds["url"], zip_path)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():  # refuse paths escaping the destination
            if not (dest / member).resolve().is_relative_to(dest.resolve()):
                sys.exit(f"Unsafe path in zip: {member}")
        zf.extractall(dest)


def source_names(dest: Path) -> list[str]:
    yamls = list(dest.glob("data.yaml")) or list(dest.rglob("data.yaml"))
    if not yamls:
        sys.exit(f"No data.yaml in {dest}")
    names = yaml.safe_load(yamls[0].read_text(encoding="utf-8"))["names"]
    return list(names) if isinstance(names, list) else [names[k] for k in sorted(names)]


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).replace("đ", "d").replace("Đ", "D"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return "".join(ch for ch in s.lower() if ch.isalnum())


def write_mapping_draft(path: Path, names: list[str]):
    lookup = {}
    for c in load_classes():
        for alias in (c.key, c.code, c.name):
            lookup.setdefault(normalize(alias), c.key)
    lines = [
        f"# {DRAFT_MARKER}: file tự tạo. Điền key (trong configs/classes.yaml) cho từng lớp nguồn,",
        "# hoặc để null để bỏ lớp đó. Kiểm tra cả các dòng đã điền sẵn.",
        f"# Xong thì XÓA dòng có chữ {DRAFT_MARKER} rồi chạy lại download_datasets.py.",
        "",
    ]
    for n in names:
        key = lookup.get(normalize(n))
        quoted = '"' + str(n).replace('"', '\\"') + '"'
        lines.append(f"{quoted}: {key}" if key else f"{quoted}: null   # TODO")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    ap.add_argument("--only", nargs="*", help="dataset names to process (default: all enabled)")
    ap.add_argument("--force", action="store_true", help="re-download even if downloads/<name> exists")
    ap.add_argument("--no-import", action="store_true", help="only download")
    args = ap.parse_args()

    datasets = yaml.safe_load(args.config.read_text(encoding="utf-8"))["datasets"]
    if args.only:
        datasets = [d for d in datasets if d["name"] in args.only]
    else:
        datasets = [d for d in datasets if d.get("enabled", True)]

    imported, pending = [], []
    for ds in datasets:
        name = ds["name"]
        dest = DOWNLOADS / name
        print(f"\n=== {name} ({ds['source']}) ===")
        if dest.exists() and args.force:
            shutil.rmtree(dest)
        if dest.exists():
            print(f"  already downloaded: {dest}")
        elif ds["source"] == "roboflow":
            download_roboflow(ds, dest)
        elif ds["source"] == "url":
            download_url(ds, dest)
        else:
            sys.exit(f"Unknown source '{ds['source']}' for {name}")

        names = source_names(dest)
        print(f"  {len(names)} classes: {names}")
        if args.no_import:
            continue

        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "import_dataset.py"),
               "--src", str(dest), "--name", name, "--unmapped", ds.get("unmapped", "error")]
        if ds.get("mapping"):
            mapping = PROJECT_ROOT / ds["mapping"]
            if not mapping.exists():
                write_mapping_draft(mapping, names)
                print(f"  -> wrote mapping draft {ds['mapping']}; fill it in, then rerun.")
                pending.append(name)
                continue
            if DRAFT_MARKER in mapping.read_text(encoding="utf-8"):
                print(f"  -> mapping {ds['mapping']} is still a draft; finish it, then rerun.")
                pending.append(name)
                continue
            cmd += ["--mapping", str(mapping)]
        if subprocess.run(cmd).returncode != 0:
            pending.append(name)
        else:
            imported.append(name)

    print("\nImported:", imported or "-")
    if pending:
        print("Needs attention:", pending)
    if imported:
        print("Next: python scripts/split_dataset.py --force")


if __name__ == "__main__":
    main()
