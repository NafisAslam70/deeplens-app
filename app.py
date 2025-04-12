import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from deeplens_engine import DeepLensFocusEngine
import av

st.set_page_config(page_title="DeepLens Engine for Focus", layout="centered")
st.title("🧠 DeepLens Engine for Focus")
st.markdown("Track your real-time focus using AI, webcam, and deep heuristics.")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

engine = DeepLensFocusEngine()

class Processor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        out_frame = engine.process_frame(img)
        return av.VideoFrame.from_ndarray(out_frame, format="bgr24")

webrtc_streamer(
    key="deeplens",
    video_processor_factory=Processor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
