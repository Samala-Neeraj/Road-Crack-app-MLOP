from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parent / "best.pt"


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


st.set_page_config(page_title="Road Crack Detection")
st.title("Road Crack Detection")
st.write("Upload a road image to detect cracks.")

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}. Place best.pt in the project root.")
    st.stop()

confidence = st.sidebar.slider(
    "Confidence threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05
)
uploaded_file = st.file_uploader(
    "Choose a road image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Run detection", type="primary"):
        try:
            model = load_model(str(MODEL_PATH))
            result = model.predict(
                source=image, conf=confidence, verbose=False
            )[0]
        except Exception as error:
            st.error(f"Prediction failed: {error}")
        else:
            original_column, result_column = st.columns(2)
            with original_column:
                st.image(image, caption="Uploaded image", use_container_width=True)
            with result_column:
                st.image(
                    result.plot()[:, :, ::-1],
                    caption="Detection result",
                    use_container_width=True,
                )

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                st.info("No cracks detected above the selected confidence threshold.")
            else:
                st.subheader(f"Detections: {len(boxes)}")
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    label = result.names[class_id]
                    score = float(box.conf[0].item())
                    st.write(f"{label}: {score:.1%}")