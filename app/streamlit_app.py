"""[Owner: TV4] Giao diện: chọn nguồn (upload/webcam/RTSP), chọn model, ngưỡng;
hiển thị video + biểu đồ score realtime + bảng sự kiện.
    streamlit run app/streamlit_app.py
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, create_video_source_track, webrtc_streamer

from app.alerts import threshold_alerts
from app.inference import SlidingWindowInference
from app.live_video import LiveVideoProcessor
from app.video_source import VideoSource
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
MODEL_OPTIONS = {
    "M4 X3D-S": ("m4_x3d.yaml", "x3d_s"),
    "M1 R(2+1)D-18": ("m1_r2plus1d.yaml", "r2plus1d_18"),
}
MAX_SCORE_HISTORY = 1000

st.set_page_config(page_title="CCTV Anomaly Detection", layout="wide")
st.title("Surveillance Camera Anomaly Detection")
st.write("AI-powered video anomaly detection system")

for key, initial in (
    ("video_source", None),
    ("temp_video_path", None),
    ("inference", None),
    ("score_rows", []),
    ("inference_error", None),
    ("last_frame_index", -1),
    ("active_source_type", None),
    ("live_video_processor", None),
    ("source_generation", 0),
    ("config_notice", None),
):
    if key not in st.session_state:
        st.session_state[key] = initial


def append_score(row):
    rows = st.session_state.score_rows
    rows.append(row)
    if len(rows) > MAX_SCORE_HISTORY:
        del rows[:-MAX_SCORE_HISTORY]


def stop_video():
    video = st.session_state.video_source
    if video is not None and not video.stop():
        return False

    processor = st.session_state.live_video_processor
    if processor is not None:
        processor.stop()

    temp_path = st.session_state.temp_video_path
    if temp_path:
        Path(temp_path).unlink(missing_ok=True)

    st.session_state.video_source = None
    st.session_state.temp_video_path = None
    st.session_state.inference = None
    st.session_state.score_rows = []
    st.session_state.inference_error = None
    st.session_state.last_frame_index = -1
    st.session_state.active_source_type = None
    st.session_state.live_video_processor = None
    st.session_state.source_generation += 1
    return True


def on_source_type_change():
    was_monitoring = st.session_state.video_source is not None
    if stop_video():
        if was_monitoring:
            st.session_state.config_notice = (
                "Input type changed; monitoring stopped. Start again to use the new source."
            )
    else:
        st.session_state.config_notice = (
            "The video reader is still stopping. Try changing the input again shortly."
        )

    if st.session_state.get("source_type") != "Upload Video":
        st.session_state.pop("uploaded_video", None)


def on_inference_config_change():
    if st.session_state.video_source is not None:
        if stop_video():
            st.session_state.config_notice = (
                "Inference settings changed; monitoring stopped. Start again to apply them."
            )
        else:
            st.session_state.config_notice = (
                "The video reader is still stopping. Try changing the settings again shortly."
            )


st.sidebar.header("Video Input")
source_type = st.sidebar.selectbox(
    "Select input type",
    ["Upload Video", "Webcam", "RTSP"],
    key="source_type",
    on_change=on_source_type_change,
)

uploaded_file = None
rtsp_url = ""
if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="uploaded_video",
    )
elif source_type == "Webcam":
    st.sidebar.write("Using the default webcam (camera 0)")
else:
    rtsp_url = st.sidebar.text_input("RTSP URL", key="rtsp_url")

model_label = st.sidebar.selectbox(
    "Model",
    list(MODEL_OPTIONS),
    key="model_label",
    on_change=on_inference_config_change,
)
mode = st.sidebar.selectbox(
    "Checkpoint mode",
    ["cls", "mil"],
    format_func=lambda value: "Classification" if value == "cls" else "MIL",
    key="mode",
    on_change=on_inference_config_change,
)
threshold = st.sidebar.slider(
    "Alert threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    key="threshold",
)
smooth_window = st.sidebar.slider(
    "Score smoothing window",
    min_value=1,
    max_value=10,
    value=1,
    key="smooth_window",
    on_change=on_inference_config_change,
)

config_name, model_name = MODEL_OPTIONS[model_label]
checkpoint_path = ROOT / "weights" / f"{model_name}_{mode}.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Device: {device}")
if not checkpoint_path.is_file():
    st.sidebar.warning(f"Checkpoint not found: {checkpoint_path}")

current_video = st.session_state.video_source
start_disabled = (
    current_video is not None
    or not checkpoint_path.is_file()
    or (source_type == "Upload Video" and uploaded_file is None)
)
start_button = st.sidebar.button(
    "Start Monitoring",
    disabled=start_disabled,
    width="stretch",
)
stop_button = st.sidebar.button(
    "Stop Monitoring",
    disabled=current_video is None,
    width="stretch",
)

if st.session_state.config_notice:
    st.sidebar.info(st.session_state.config_notice)
    st.session_state.config_notice = None

if start_button:
    source = None
    temp_path = None
    video = None
    live_processor = None
    try:
        if source_type == "Upload Video":
            if uploaded_file is None:
                st.sidebar.error("Please upload a video first.")
            else:
                suffix = Path(uploaded_file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                source = temp_path
        elif source_type == "Webcam":
            source = 0
        elif rtsp_url.strip():
            source = rtsp_url.strip()
        else:
            st.sidebar.error("Please enter an RTSP URL.")

        if source is not None:
            with st.spinner("Loading model and opening video source..."):
                cfg = load_config(ROOT / "configs" / config_name)
                inference = SlidingWindowInference(
                    cfg,
                    checkpoint_path,
                    mode=mode,
                    device=device,
                    smooth_window=smooth_window,
                )
                video = VideoSource(source).start()

            st.session_state.video_source = video
            st.session_state.temp_video_path = temp_path
            st.session_state.inference = inference
            st.session_state.score_rows = []
            st.session_state.inference_error = None
            st.session_state.last_frame_index = -1
            st.session_state.active_source_type = source_type
            live_processor = (
                LiveVideoProcessor(video, inference)
                if source_type in {"Webcam", "RTSP"}
                else None
            )
            st.session_state.live_video_processor = live_processor
            st.session_state.config_notice = None
            st.rerun()
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        if live_processor is not None:
            live_processor.stop()
        if video is not None:
            video.stop()
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        st.sidebar.error(f"Cannot start monitoring: {exc}")

if stop_button:
    if stop_video():
        st.rerun()
    else:
        st.sidebar.warning("The video reader is still stopping. Try again shortly.")


st.subheader("Video Monitoring")
video_column, info_column = st.columns([2, 1])

with video_column:
    st.subheader("Live Video")
    if source_type == "Upload Video" and uploaded_file is not None:
        st.video(uploaded_file.getvalue())
    elif (
        source_type in {"Webcam", "RTSP"}
        and st.session_state.video_source is not None
        and st.session_state.live_video_processor is not None
    ):
        live_processor = st.session_state.live_video_processor
        source_track = create_video_source_track(
            callback=live_processor.next_frame,
            key=f"live-source-{st.session_state.source_generation}",
            fps=30,
        )
        webrtc_streamer(
            key=f"live-preview-{st.session_state.source_generation}",
            mode=WebRtcMode.RECVONLY,
            source_video_track=source_track,
            desired_playing_state=True,
            media_stream_constraints={"video": True, "audio": False},
            media_toggle_controls=False,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )
    else:
        st.info("Select a video source and start monitoring.")

info_placeholder = info_column.empty()
scores_placeholder = st.empty()
events_placeholder = st.empty()


def process_uploaded_frames(video):
    inference = st.session_state.inference
    for frame_number in range(64):
        success, frame, frame_index, timestamp = video.read_with_metadata(
            timeout=0.01 if frame_number == 0 else 0
        )
        if not success:
            break
        if frame_index is None:
            frame_index = st.session_state.get("last_frame_index", -1) + 1
            timestamp = frame_index / video.fps
        st.session_state.last_frame_index = frame_index
        if inference is not None and st.session_state.inference_error is None:
            try:
                row = inference.process_frame(frame, frame_index, timestamp)
            except (RuntimeError, ValueError) as exc:
                st.session_state.inference_error = str(exc)
            else:
                if row is not None:
                    append_score(row)


def drain_live_scores():
    processor = st.session_state.live_video_processor
    if processor is None:
        return
    for row in processor.drain_scores():
        append_score(row)
    processor_error = processor.get_error()
    if processor_error:
        st.session_state.inference_error = processor_error


@st.fragment(run_every=0.5)
def display_monitoring():
    video = st.session_state.video_source
    if video is not None:
        if st.session_state.active_source_type == "Upload Video":
            process_uploaded_frames(video)
        else:
            drain_live_scores()

    rows = st.session_state.score_rows
    score_frame = pd.DataFrame(rows)
    latest = rows[-1] if rows else None

    with info_placeholder.container():
        st.subheader("Detection Status")
        if video is None:
            st.warning("No video source connected")
        elif video.error:
            st.error("Video source error")
        elif video.is_ended() and video.frames.empty():
            st.info("Video source finished")
        else:
            st.success("Video source connected")

        if st.session_state.inference_error:
            st.error(f"Inference failed: {st.session_state.inference_error}")
        elif video is not None and video.error:
            st.error(f"Video error: {video.error}")
        elif video is not None and video.is_ended() and video.frames.empty():
            st.warning("The video has ended.")

        st.subheader("Anomaly Detection")
        if latest is None:
            st.metric("Anomaly Score", "Waiting for clip")
        else:
            st.metric("Anomaly Score", f"{latest['score']:.3f}")
            if latest["score"] >= threshold:
                st.warning("Alert: score is at or above the threshold.")
            else:
                st.success("Score is below the alert threshold.")
            if mode == "cls":
                st.write(f"Predicted class: {latest['predicted_class']}")
                st.write(f"Class confidence: {latest['class_confidence']:.3f}")

        st.subheader("System Information")
        st.write(f"Input type: {source_type}")
        st.write(f"Model: {model_label}")
        st.write(f"Checkpoint mode: {mode}")
        st.write(f"Checkpoint: {checkpoint_path.relative_to(ROOT)}")
        st.write(f"Checkpoint available: {'yes' if checkpoint_path.is_file() else 'no'}")
        st.write(f"Device: {device}")
        st.write(f"Alert threshold: {threshold:.2f}")
        st.write(f"Score smoothing window: {smooth_window}")
        st.write(f"Scores retained: {len(rows)}")

    with scores_placeholder.container():
        st.subheader("Anomaly Score Over Time")
        if score_frame.empty:
            st.info("Scores will appear when a complete clip has been sampled.")
        else:
            st.line_chart(score_frame.set_index("timestamp_seconds")[["score"]])
            st.download_button(
                "Download score CSV",
                data=score_frame.to_csv(index=False).encode("utf-8"),
                file_name="anomaly_scores.csv",
                mime="text/csv",
            )

    with events_placeholder.container():
        st.subheader("Detected Events")
        if score_frame.empty:
            st.info("No scored clips yet.")
        else:
            events = threshold_alerts(score_frame, threshold)
            if events.empty:
                st.success("No clips have crossed the alert threshold.")
            else:
                st.warning(f"{len(events)} clip(s) crossed the alert threshold.")
                columns = ["timestamp_seconds", "score"]
                if mode == "cls":
                    columns.append("predicted_class")
                st.dataframe(events[columns], width="stretch", hide_index=True)
            st.caption("Threshold alerts are for demonstration and require human review.")


display_monitoring()
