# 🧠 DeepLens Engine for Focus

A real-time attention tracker using YOLOv11-cls, yolov8n + MediaPipe + Streamlit.

## 🔍 Features
- Focus detection every second using YOLO classification
- Gaze tracking (LookingAway override)
- Person presence check (Absent override)
- 30s rolling buffer with 20/30 heuristic
- Live status + distraction logging

## 🖥️ Run Locally

```bash
streamlit run app.py
