"""[Owner: TV4] Giao diện: chọn nguồn (upload/webcam/RTSP), chọn model, ngưỡng;
hiển thị video + biểu đồ score realtime + bảng sự kiện.
    streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="CCTV Anomaly Detection", layout="wide")
st.title("Phát hiện bạo lực & bất thường từ camera giám sát")
st.info("TODO(TV4): tích hợp app.inference + app.alerts")
