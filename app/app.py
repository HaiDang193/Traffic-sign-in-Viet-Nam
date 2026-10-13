"""Streamlit demo: Vietnamese traffic sign detection on images and road videos.

Run from the project root:
    streamlit run app/app.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vnsign.inference import SignDetector, SignLogger, draw_detections  # noqa: E402

st.set_page_config(page_title="Nhận diện biển báo giao thông VN", page_icon="🚦", layout="wide")


@st.cache_resource
def get_detector(weights: str) -> SignDetector:
    return SignDetector(weights)


def find_weights() -> list[str]:
    found = sorted((ROOT / "models").glob("*.pt")) + sorted((ROOT / "runs").glob("*/weights/best.pt"))
    return [str(p.relative_to(ROOT)) for p in found]


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Cấu hình")
weights_list = find_weights()
if not weights_list:
    st.sidebar.error("Chưa có model. Đặt file .pt vào `models/` hoặc train để có `runs/*/weights/best.pt`.")
    st.stop()
weights = st.sidebar.selectbox("Model", weights_list)
conf = st.sidebar.slider("Ngưỡng tin cậy (conf)", 0.05, 0.95, 0.40, 0.05)
imgsz = st.sidebar.select_slider("Kích thước ảnh vào (imgsz)", [640, 800, 960, 1280], value=960)
min_hits = st.sidebar.slider("Video: số frame tối thiểu để ghi nhận 1 biển", 1, 10, 3)

det = get_detector(str(ROOT / weights))
det.conf, det.imgsz = conf, imgsz

st.title("🚦 Nhận diện biển báo giao thông Việt Nam")
tab_img, tab_video = st.tabs(["🖼️ Ảnh", "🎥 Video hành trình"])

# ---------------- Image ----------------
with tab_img:
    files = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    tiled = st.checkbox("Chế độ cắt ô (bắt biển nhỏ / ở xa, chậm hơn)")
    for f in files or []:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        dets = det.predict_tiled(img) if tiled else det.predict(img)
        col1, col2 = st.columns([3, 2])
        col1.image(cv2.cvtColor(draw_detections(img, dets), cv2.COLOR_BGR2RGB), caption=f.name,
                   use_container_width=True)
        with col2:
            st.subheader(f"Phát hiện {len(dets)} biển báo")
            for d in sorted(dets, key=lambda d: -d.conf):
                x1, y1, x2, y2 = d.xyxy
                c1, c2 = st.columns([1, 4])
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size:
                    c1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=64)
                c2.markdown(f"**{d.cls.code} – {d.cls.name}** ({d.conf:.0%})  \n{d.cls.meaning}")

# ---------------- Video ----------------
with tab_video:
    vf = st.file_uploader("Chọn video quay từ xe (mp4, mov, avi)", type=["mp4", "mov", "avi", "mkv"])
    if vf and st.button("▶️ Xử lý video", type="primary"):
        tmp = Path(tempfile.mkdtemp())
        src = tmp / vf.name
        src.write_bytes(vf.read())
        out_path = tmp / "annotated.mp4"

        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        logger = SignLogger(fps, min_hits=min_hits)
        det.reset_tracker()
        # H.264 via ffmpeg so the result plays in the browser (OpenCV's mp4v does not).
        # macro_block_size=2: libx264 needs even width/height, imageio pads odd sizes.
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7, macro_block_size=2)

        progress = st.progress(0.0, text="Đang xử lý...")
        preview = st.empty()
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = det.track(frame)
            logger.update(idx, frame, dets)
            annotated = cv2.cvtColor(draw_detections(frame, dets), cv2.COLOR_BGR2RGB)
            writer.append_data(annotated)
            if idx % 15 == 0:
                preview.image(annotated, use_container_width=True)
                progress.progress(min(idx / n_frames, 1.0), text=f"Frame {idx}/{n_frames}")
            idx += 1
        cap.release()
        writer.close()
        progress.progress(1.0, text="Xong!")
        preview.empty()
        st.session_state["video_result"] = (out_path.read_bytes(), logger.rows(),
                                            {r["track_id"]: logger.crop(r["track_id"]) for r in logger.rows()})

    if "video_result" in st.session_state:
        video_bytes, rows, crops = st.session_state["video_result"]
        col1, col2 = st.columns([3, 2])
        col1.video(video_bytes)
        col1.download_button("⬇️ Tải video kết quả", video_bytes, "annotated.mp4", "video/mp4")
        with col2:
            st.subheader(f"Nhật ký hành trình: {len(rows)} biển")
            if rows:
                df = pd.DataFrame(rows)[["time", "code", "name", "confidence", "frames_seen", "meaning"]]
                st.download_button("⬇️ Tải CSV", df.to_csv(index=False).encode("utf-8-sig"),
                                   "signs_log.csv", "text/csv")
                for r in rows:
                    c1, c2 = st.columns([1, 4])
                    crop = crops.get(r["track_id"])
                    if crop is not None and crop.size:
                        c1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=64)
                    c2.markdown(f"`{r['time']}` **{r['code']} – {r['name']}** ({r['confidence']:.0%})  \n"
                                f"{r['meaning']}")
