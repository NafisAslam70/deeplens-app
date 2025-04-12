import cv2
import time
from datetime import datetime
import torch
from PIL import Image
import torchvision.transforms as transforms
import mediapipe as mp
from collections import deque

class DeepLensFocusEngine:
    def __init__(self):
        self.classifier_model = torch.jit.load("weights/best.torchscript")
        self.classifier_model.eval()
        self.classifier_names = {
            0: "Focused",
            1: "LookingAway",
            2: "Phone",
            3: "Absent",
            4: "BadPosture",
            5: "Drowsy"
        }

        from ultralytics import YOLO
        self.detection_model = YOLO("yolov8n.pt", task='detect')

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        self.secondWindow = deque(maxlen=30)
        self.focusLog = []
        self.focusSeconds = 0
        self.distractedSeconds = 0
        self.last_logged_time = time.time()
        self.last_override_message = ""
        self.last_override_time = 0
        self.last_final_decision = "Waiting..."
        self.last_final_reason = "-"
        self.live_status_label = "Waiting..."
        self.live_status_conf = 0.0
        self.rolling_summary_msg = "-"
        self.summary_inserted_msg = ""
        self.summary_inserted_time = 0

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def gaze_is_away(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                lm = face.landmark
                left_eye = lm[33]
                right_eye = lm[133]
                iris = lm[468]
                width = right_eye.x - left_eye.x
                if width == 0: return False
                iris_pos = (iris.x - left_eye.x) / width
                return not (0.35 <= iris_pos <= 0.65)
        return False

    def process_frame(self, frame):
        # Preprocess and infer
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.classifier_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            class_id = torch.argmax(probs).item()
            confidence = probs[class_id].item() * 100
            label = self.classifier_names.get(class_id, "-")

        self.live_status_label = label
        self.live_status_conf = confidence

        override_message = ""
        if label == "LookingAway":
            if not self.gaze_is_away(frame):
                label = "Focused"
                override_message = "🔎 Gaze corrected → LookingCenter ✅"
            else:
                override_message = "🔎 Gaze confirmed → LookingAway ❌"
        elif label == "Absent":
            det_results = self.detection_model(frame)
            detected_classes = det_results[0].boxes.cls.cpu().numpy().astype(int)
            if 0 in detected_classes:
                label = "Focused"
                override_message = "🙋 Person detected → Overridden to Focused"
            else:
                override_message = "🙋 No person → Absent confirmed ✅"
        elif label == "Focused":
            override_message = "✅ You're focused. No auxiliary validation needed."

        if override_message:
            self.last_override_message = override_message
            self.last_override_time = time.time()
        elif time.time() - self.last_override_time > 5:
            self.last_override_message = ""

        focusState = "Focused"
        reason = None
        if label != "Focused":
            focusState = "Distracted"
            reason = label

        self.secondWindow.append({"focusState": focusState, "reason": reason, "confidence": confidence})

        if len(self.secondWindow) == 30 and time.time() - self.last_logged_time >= 30:
            self.last_logged_time = time.time()
            focused_count = sum(1 for e in self.secondWindow if e["focusState"] == "Focused")
            distracted_count = 30 - focused_count

            if focused_count >= 20:
                self.last_final_decision = "FOCUSED ✅"
                self.last_final_reason = "-"
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "focusState": "Focused",
                    "reason": None,
                    "confidence": round(sum(e["confidence"] for e in self.secondWindow) / 30, 2)
                }
                self.focusSeconds += 30
            else:
                reasons = [e["reason"] for e in self.secondWindow if e["reason"]]
                top_reason = max(set(reasons), key=reasons.count) if reasons else "-"
                self.last_final_decision = f"DISTRACTED ❌ — Reason: {top_reason}"
                self.last_final_reason = top_reason
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "focusState": "Distracted",
                    "reason": top_reason,
                    "confidence": round(sum(e["confidence"] for e in self.secondWindow if e["focusState"] == "Distracted") / distracted_count, 2)
                }
                self.distractedSeconds += 30

            self.focusLog.append(entry)
            self.secondWindow.clear()

            self.summary_inserted_msg = f"✅ 30s summary inserted: {entry['focusState']}"
            self.summary_inserted_time = time.time()
            self.rolling_summary_msg = f"Last 30s → {entry['focusState']} ({focused_count}/30 Focused) | Confidence: {entry['confidence']}%"
            if entry['reason']:
                self.rolling_summary_msg += f" | Reason: {entry['reason']}"

        if self.summary_inserted_msg and time.time() - self.summary_inserted_time > 10:
            self.summary_inserted_msg = ""

        return self.overlay_ui_elements(frame, label, confidence)

    def overlay_ui_elements(self, frame, label, confidence):
        # Top status
        cv2.putText(frame, f"🌟 LIVE STATUS: {self.live_status_label} ({self.live_status_conf:.0f}%)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"📊 FINAL STATUS: {self.last_final_decision}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if "FOCUSED" in self.last_final_decision else (0, 0, 255), 2)

        box_x, box_y = 10, frame.shape[0] - 180
        box_width, box_height = 360, 180
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (20, 20, 20), -1)
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (80, 80, 80), 2)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "🧠 DeepLens Engine For Focus", (box_x + 12, box_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"🔹 Predicted: {label} ({confidence:.0f}%)", (box_x + 10, box_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if label == "Focused" else (255, 0, 0), 1)

        if self.last_override_message:
            cv2.putText(frame, self.last_override_message, (box_x + 10, box_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)

        cv2.putText(frame, "🔁 Rolling 30s:", (box_x + 10, box_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for i, entry in enumerate(self.secondWindow):
            color = (0, 255, 0) if entry["focusState"] == "Focused" else (0, 0, 255)
            x = box_x + 140 + i * 6
            y = box_y + 78
            cv2.rectangle(frame, (x, y), (x + 5, y + 12), color, -1)

        words = self.rolling_summary_msg.split()
        line1, line2 = "", ""
        for word in words:
            test_line = line1 + word + " "
            if cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] < box_width - 20:
                line1 += word + " "
            else:
                line2 += word + " "
        cv2.putText(frame, line1.strip(), (box_x + 10, box_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)
        if line2:
            cv2.putText(frame, line2.strip(), (box_x + 10, box_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)

        if self.summary_inserted_msg:
            cv2.putText(frame, self.summary_inserted_msg, (box_x + 10, box_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.putText(frame, "📊 Last 5 min:", (box_x + 10, box_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for i, log in enumerate(self.focusLog[-10:]):
            color = (0, 255, 0) if log["focusState"] == "Focused" else (0, 0, 255)
            x = box_x + 140 + i * 16
            y = box_y + 137
            cv2.rectangle(frame, (x, y), (x + 14, y + 14), color, -1)

        cv2.putText(frame, f"⏱ Focused: {self.focusSeconds}s", (box_x + 10, box_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"⛔ Distracted: {self.distractedSeconds}s", (box_x + 180, box_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "🌟 Stay present. Stay powerful.", (box_x + 10, box_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)

        return frame
