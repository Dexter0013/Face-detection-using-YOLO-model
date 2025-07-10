import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2 # Import cv2 for converting AvFrame to numpy array

# Load YOLO11 Nano model
# Make sure this path is correct and the model exists after training
model = YOLO("/content/optimized/runs/yolo11_optimized_run3/weights/best.pt")

# Define frame processor
class YOLOTransformer(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference
        results = model(img)

        # Annotated frame
        annotated_frame = results[0].plot()

        # Convert the annotated frame back to an AvFrame
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Streamlit UI
st.title("YOLO11n + WebRTC Real-Time Object Detection")
st.markdown("This app uses your webcam to detect objects live using YOLO11 Nano.")

# Start WebRTC stream
webrtc_streamer(
    key="yolo11n-stream",
    video_transformer_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)