from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.alerts import threshold_alerts

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_threshold_alerts_selects_rows_at_or_above_threshold():
    scores = pd.DataFrame(
        {"timestamp_seconds": [0.0, 1.0, 2.0], "score": [0.2, 0.7, 0.9]}
    )

    events = threshold_alerts(scores, 0.7)

    assert events["timestamp_seconds"].tolist() == [1.0, 2.0]
    assert scores["score"].tolist() == [0.2, 0.7, 0.9]


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_threshold_alerts_rejects_out_of_range_threshold(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        threshold_alerts(pd.DataFrame({"score": [0.5]}), threshold)


def test_upload_control_tracks_selected_input_type():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert len(app.get("file_uploader")) == 1

    app.selectbox[0].set_value("Webcam").run()
    assert len(app.get("file_uploader")) == 0
    assert len(app.get("text_input")) == 0

    app.selectbox[0].set_value("RTSP").run()
    assert len(app.get("file_uploader")) == 0
    assert len(app.get("text_input")) == 1

    app.selectbox[0].set_value("Upload Video").run()
    assert len(app.get("file_uploader")) == 1
    assert len(app.get("text_input")) == 0


def test_system_information_tracks_model_and_checkpoint_mode():
    app = AppTest.from_file(str(APP_PATH)).run()

    app.selectbox[1].set_value("M1 R(2+1)D-18").run()
    app.selectbox[2].set_value("mil").run()

    markdown = [element.value for element in app.get("markdown")]
    assert "Model: M1 R(2+1)D-18" in markdown
    assert "Checkpoint mode: mil" in markdown
