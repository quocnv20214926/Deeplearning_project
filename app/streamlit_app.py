"""[Owner: TV4] Giao diện: chọn nguồn (upload/webcam/RTSP), chọn model, ngưỡng;
hiển thị video + biểu đồ score realtime + bảng sự kiện.
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import cv2
import tempfile
import os

from app.video_source import VideoSource


st.set_page_config(
    page_title="CCTV Anomaly Detection",
    layout="wide"
)

st.title("Surveillance Camera Anomaly Detection")
st.write("AI-powered video anomaly detection system")


# -----------------------------------
# Initialize session state
# -----------------------------------

if "video_source" not in st.session_state:
    st.session_state.video_source = None

if "temp_video_path" not in st.session_state:
    st.session_state.temp_video_path = None

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None


# -----------------------------------
# Stop video function
# -----------------------------------

def stop_video():

    video = st.session_state.video_source

    if video is None:
        return True

    stopped = video.stop()

    if stopped:

        st.session_state.video_source = None
        st.session_state.last_frame = None

        temp_path = st.session_state.temp_video_path

        if temp_path is not None:
            if os.path.exists(temp_path):
                os.remove(temp_path)

            st.session_state.temp_video_path = None

    return stopped


# -----------------------------------
# Sidebar: Video input
# -----------------------------------

st.sidebar.header("Video Input")

source_type = st.sidebar.selectbox(
    "Select input type",
    ["Upload Video", "Webcam", "RTSP"]
)

uploaded_file = None
rtsp_url = ""

if source_type == "Upload Video":

    uploaded_file = st.sidebar.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"]
    )

elif source_type == "Webcam":

    st.sidebar.write("Using the default webcam (camera 0)")

else:

    rtsp_url = st.sidebar.text_input("RTSP URL")


# -----------------------------------
# Start and Stop buttons
# -----------------------------------

current_video = st.session_state.video_source

start_button = st.sidebar.button(
    "Start Monitoring",
    disabled=current_video is not None,
    width='stretch'
)

stop_button = st.sidebar.button(
    "Stop Monitoring",
    disabled=current_video is None,
    width='stretch'
)


if start_button:

    source = None
    temp_path = None

    try:

        if source_type == "Upload Video":

            if uploaded_file is None:
                st.sidebar.error("Please upload a video first.")

            else:

                # Save uploaded video to a temporary file.
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                source = temp_path

        elif source_type == "Webcam":

            source = 0

        else:

            if rtsp_url.strip() == "":
                st.sidebar.error("Please enter an RTSP URL.")

            else:
                source = rtsp_url.strip()

        if source is not None:

            video = VideoSource(source)
            video.start()

            st.session_state.video_source = video
            st.session_state.temp_video_path = temp_path
            st.session_state.last_frame = None

            st.rerun()

    except Exception as e:

        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)

        st.sidebar.error(f"Cannot start video: {e}")


if stop_button:

    if stop_video():
        st.rerun()

    else:
        st.sidebar.warning(
            "The video reader is still stopping. Try again shortly."
        )


# -----------------------------------
# Main video area
# -----------------------------------

st.subheader("Video Monitoring")

video_column, info_column = st.columns([2, 1])

with video_column:

    st.subheader("Live Video")

    @st.fragment(run_every=0.1)
    def display_video():

        video = st.session_state.video_source

        if video is None:
            st.info("Select a video source and start monitoring.")
            return

        success, frame = video.read(timeout=0.05)

        if success:

            # OpenCV uses BGR, while Streamlit displays RGB.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            st.session_state.last_frame = frame

        if st.session_state.last_frame is not None:

            st.image(
                st.session_state.last_frame,
                channels="RGB",
                width='stretch'
            )

        elif video.error:

            st.error(video.error)

        else:

            st.info("Waiting for video frames...")

        if video.error:
            st.error(f"Video error: {video.error}")

        elif video.is_ended() and video.frames.empty():
            st.warning("Video has ended.")

    display_video()


# -----------------------------------
# Detection information
# -----------------------------------

with info_column:

    st.subheader("Detection Status")

    if st.session_state.video_source is not None:

        st.success("Video source connected")

        st.write("Source is running or has finished reading.")

    else:

        st.warning("No video source connected")

    st.subheader("Anomaly Detection")

    st.metric("Anomaly Score", "Not available")
    st.info("Inference module is not connected yet.")

    st.subheader("System Information")

    st.write("Model: Not connected")
    st.write("FPS: Not measured")


# -----------------------------------
# Anomaly score chart
# -----------------------------------

st.subheader("Anomaly Score Over Time")

st.info(
    "The anomaly score chart will be connected "
    "after the inference module is implemented."
)


# -----------------------------------
# Event history
# -----------------------------------

st.subheader("Detected Events")

st.info(
    "Detected events will appear here after "
    "the anomaly detection module is implemented."
)
