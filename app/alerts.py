"""[Owner: TV4] Ngưỡng + hysteresis (>= N snippet liên tiếp) -> sự kiện; lưu ảnh + clip ngắn,
ghi SQLite (results/events.db); tuỳ chọn gửi Telegram (token trong .env)."""

import pandas as pd

def threshold_alerts(scores, threshold):
    """Return score rows at or above a probability threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if "score" not in scores.columns:
        raise ValueError("scores must contain a 'score' column")
    return scores.loc[scores["score"] >= threshold].copy()
