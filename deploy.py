import streamlit as st
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def save_uploaded_file(uploaded_file):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "uploaded_video.mp4"

def perform_object_detection(video_file, output_file):
    model = get_model()
    model.eval()

    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    detected_frames = []

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = transform_frame(frame)
            output = model([input_tensor])

            for box, score, label in zip(output[0]['boxes'], output[0]['scores'], output[0]['labels']):
                if score > 0.5:  # Filter out low-confidence detections
                    xmin, ymin, xmax, ymax = box.cpu().numpy().astype(int)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"Class: {label}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            detected_frames.append(frame)

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(frame_pil, channels="RGB", use_column_width=True)

    cap.release()
    out.release()

    if len(detected_frames) > 0:
        detected_output_file = "detected_frames.mp4"
        out = cv2.VideoWriter(detected_output_file, fourcc, fps, (width, height))
        for frame in detected_frames:
            out.write(frame)
        out.release()
        st.success(f"Detected frames saved as {detected_output_file}")

def get_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 91  # COCO dataset has 91 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def transform_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(frame)
    return input_tensor

def main():
    st.title("Violence Detection on Video")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        video_file_path = save_uploaded_file(uploaded_file)
        st.success(f"Uploaded video saved as {video_file_path}")

        output_file = "output.mp4"
        perform_object_detection(video_file_path, output_file)
        st.success(f"Processed video saved as {output_file}")
        with open('detected_frames.mp4', 'rb') as f:
            st.download_button('Download Zip', f, file_name='video.mp4')

if __name__ == "__main__":
    main()