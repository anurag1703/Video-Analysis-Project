import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import os

def select_frames(video_path, num_frames=20, threshold=25):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= num_frames:
        return list(range(total_frames))

    uniform_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    selected_frames = list(uniform_indices)

    prev_gray = None
    for frame_num in selected_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                frame_diff = cv2.absdiff(prev_gray, gray)
                diff_mean = np.mean(frame_diff)
                if diff_mean > threshold:
                    if frame_num not in selected_frames:
                        selected_frames.append(frame_num)
            prev_gray = gray
    video.release()
    return selected_frames

def detect_objects(video_path, selected_frames, model_path="yolov8n.pt", confidence_threshold=0.3, iou_threshold=0.45):
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_path)
    detected_items = set()
    results = []
    for frame_num in selected_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        if ret:
            results_frame = model(frame, conf=confidence_threshold, iou=iou_threshold)[0]
            if len(results_frame.boxes) > 0:
                for i, (*box, conf, cls) in enumerate(results_frame.boxes.data):
                    class_name = results_frame.names[int(cls)]
                    if class_name in ['person', 'clothing', 'dress', 'shirt', 'pants', 'shorts', 'skirt', 'shoes']:
                        item_key = (class_name, tuple(map(int, box)))
                        if item_key not in detected_items:
                            detected_items.add(item_key)
                            x1, y1, x2, y2 = map(int, box)
                            cropped_image = frame[y1:y2, x1:x2]
                            if cropped_image.size > 0:
                                _, encoded_image = cv2.imencode(".jpg", cropped_image)
                                image_bytes = encoded_image.tobytes()
                                results.append((image_bytes, class_name, conf))
    video.release()
    return results

st.title("Shoppable Item Extractor")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov"])

if uploaded_video:
    video_bytes = uploaded_video.getvalue()
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    selected_frames = select_frames(video_path)
    detection_results = detect_objects(video_path, selected_frames)

    if detection_results:
        st.write("Extracted Items:")
        for image_bytes, class_name, conf in detection_results:
            st.image(image_bytes, caption=f"Class: {class_name}, Confidence: {conf:.2f}", use_column_width=True)
    else:
        st.write("No clothing items detected in the video.")
    os.remove(video_path)