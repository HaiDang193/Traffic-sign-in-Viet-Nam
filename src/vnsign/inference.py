"""Inference helpers: run YOLO on images / video frames, draw Vietnamese labels, log signs seen in a video."""
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .classes import SignClass, load_classes

# Box colors per sign group, in RGB (boxes are drawn with PIL).
GROUP_COLORS = {
    "cam": (220, 30, 30),
    "nguy_hiem": (245, 180, 0),
    "hieu_lenh": (30, 90, 220),
    "chi_dan": (0, 150, 120),
    "phu": (130, 130, 130),
    "khac": (160, 60, 200),
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


@dataclass
class Detection:
    cls: SignClass
    conf: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None

    def to_dict(self) -> dict:
        return {"code": self.cls.code, "key": self.cls.key, "name": self.cls.name,
                "confidence": round(self.conf, 4), "bbox": list(self.xyxy), "track_id": self.track_id}


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)  # may lack Vietnamese glyphs


class SignDetector:
    def __init__(self, weights: str | Path, conf: float = 0.4, imgsz: int = 960, device=None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        classes = {c.key: c for c in load_classes()}
        # Model names come from data.yaml (class keys). Unknown names get a placeholder class.
        self.id_to_class = {
            i: classes.get(name, SignClass(i, name, name, "khac", name, ""))
            for i, name in self.model.names.items()
        }

    def _to_detections(self, result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        return [
            Detection(self.id_to_class[int(c)], float(p), tuple(int(v) for v in xyxy), tid)
            for c, p, xyxy, tid in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), ids)
        ]

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(image_bgr, conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        return self._to_detections(result)

    def predict_tiled(self, image_bgr: np.ndarray, tile: int = 640, overlap: float = 0.25,
                      iou: float = 0.5) -> list[Detection]:
        """Sliced inference (same idea as SAHI) for small, far-away signs.

        Runs the model on overlapping tiles at native resolution plus on the whole image,
        then merges everything with class-wise NMS. Slower; use for images, not video.
        """
        import torch
        from torchvision.ops import batched_nms

        h, w = image_bgr.shape[:2]
        step = max(1, int(tile * (1 - overlap)))
        xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
        ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
        if xs[-1] + tile < w:
            xs.append(w - tile)
        if ys[-1] + tile < h:
            ys.append(h - tile)
        offsets = [(max(x, 0), max(y, 0)) for y in ys for x in xs]
        crops = [image_bgr[y:y + tile, x:x + tile] for x, y in offsets]

        boxes, scores, classes = [], [], []
        results = self.model.predict(crops, conf=self.conf, imgsz=tile, device=self.device, verbose=False)
        results += self.model.predict(image_bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        for (ox, oy), r in zip(offsets + [(0, 0)], results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().clone()
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
            boxes.append(b)
            scores.append(r.boxes.conf.cpu())
            classes.append(r.boxes.cls.cpu())
        if not boxes:
            return []
        boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
        keep = batched_nms(boxes, scores, classes.long(), iou).tolist()
        # A sign cut by a tile border gives a partial box that NMS (IoU) does not remove:
        # also drop boxes mostly inside a higher-scoring box (intersection / smaller area).
        final = []
        for i in keep:  # keep is sorted by score, descending
            bi = boxes[i]
            area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
            dup = False
            for j in final:
                bj = boxes[j]
                iw = (torch.min(bi[2], bj[2]) - torch.max(bi[0], bj[0])).clamp(min=0)
                ih = (torch.min(bi[3], bj[3]) - torch.max(bi[1], bj[1])).clamp(min=0)
                smaller = torch.min(area_i, (bj[2] - bj[0]) * (bj[3] - bj[1]))
                if smaller > 0 and iw * ih / smaller > 0.6:
                    dup = True
                    break
            if not dup:
                final.append(i)
        return [Detection(self.id_to_class[int(classes[i])], float(scores[i]),
                          tuple(int(v) for v in boxes[i].tolist())) for i in final]

    def track(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect + track on one video frame. Call frames in order; tracker state persists."""
        result = self.model.track(frame_bgr, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                  persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        return self._to_detections(result)

    def reset_tracker(self):
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for t in predictor.trackers:
                t.reset()


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw boxes with Vietnamese labels (OpenCV putText cannot render Vietnamese, so use PIL)."""
    h = image_bgr.shape[0]
    font = load_font(max(14, h // 45))
    width = max(2, h // 360)
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for d in detections:
        color = GROUP_COLORS.get(d.cls.group, GROUP_COLORS["khac"])
        x1, y1, x2, y2 = d.xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        label = f"{d.cls.code} {d.cls.name} {d.conf:.2f}"
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx2 - tx1, ty2 - ty1 + 6
        ty = y1 - th if y1 - th >= 0 else y2
        draw.rectangle((x1, ty, x1 + tw + 8, ty + th), fill=color)
        draw.text((x1 + 4, ty + 2 - ty1), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@dataclass
class _Track:
    first_frame: int
    last_frame: int
    hits: int = 0
    votes: dict = field(default_factory=lambda: defaultdict(float))
    best_conf: float = 0.0
    best_crop: np.ndarray | None = None


class SignLogger:
    """Collects one log entry per tracked sign (instead of one per frame).

    A track is reported only if it was seen in >= min_hits frames, which filters
    out one-frame false positives. The class is a confidence-weighted vote over frames.
    """

    def __init__(self, fps: float, min_hits: int = 3):
        self.fps = fps or 30.0
        self.min_hits = min_hits
        self.tracks: dict[int, _Track] = {}
        self.classes: dict[str, SignClass] = {}

    def update(self, frame_idx: int, frame_bgr: np.ndarray, detections: list[Detection]):
        for d in detections:
            if d.track_id is None:
                continue
            t = self.tracks.setdefault(d.track_id, _Track(frame_idx, frame_idx))
            t.last_frame = frame_idx
            t.hits += 1
            t.votes[d.cls.key] += d.conf
            self.classes[d.cls.key] = d.cls
            if d.conf > t.best_conf:
                x1, y1, x2, y2 = d.xyxy
                t.best_conf = d.conf
                t.best_crop = frame_bgr[max(0, y1):y2, max(0, x1):x2].copy()

    def rows(self) -> list[dict]:
        out = []
        for tid, t in sorted(self.tracks.items(), key=lambda kv: kv[1].first_frame):
            if t.hits < self.min_hits:
                continue
            cls = self.classes[max(t.votes, key=t.votes.get)]
            sec = t.first_frame / self.fps
            out.append({
                "time": f"{int(sec // 60):02d}:{sec % 60:05.2f}",
                "time_sec": round(sec, 2),
                "track_id": tid,
                "code": cls.code,
                "key": cls.key,
                "name": cls.name,
                "meaning": cls.meaning,
                "confidence": round(t.best_conf, 4),
                "frames_seen": t.hits,
            })
        return out

    def crop(self, track_id: int) -> np.ndarray | None:
        t = self.tracks.get(track_id)
        return None if t is None else t.best_crop
