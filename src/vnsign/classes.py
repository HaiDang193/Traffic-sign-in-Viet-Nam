"""Load the project's class list (configs/classes.yaml)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSES_FILE = PROJECT_ROOT / "configs" / "classes.yaml"


@dataclass(frozen=True)
class SignClass:
    id: int
    key: str
    code: str
    group: str
    name: str
    meaning: str


def load_classes(path: Path = CLASSES_FILE) -> list[SignClass]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["classes"]
    classes = [SignClass(id=i, **{k: str(c[k]) for k in ("key", "code", "group", "name", "meaning")})
               for i, c in enumerate(raw)]
    keys = [c.key for c in classes]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"Duplicate class keys in {path}: {sorted(dupes)}")
    return classes


def key_to_id(classes: list[SignClass]) -> dict[str, int]:
    return {c.key: c.id for c in classes}
