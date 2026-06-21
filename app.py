
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from disease_info import disease_info

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.success("""
🌱 Plant Disease Detection

Dataset: PlantVillage

Validation Accuracy: 94.69%

Number of Classes: 5

Model: CNN
""")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "plant_disease_model.h5"
    )

model = load_model()

# --------------------------------------------------
# CLASSES
# --------------------------------------------------

classes = [
    "Potato Early Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌱 Plant Disease Detection Using Deep Learning")

st.markdown("""
Upload a leaf image or capture a photo to identify plant diseases using a trained CNN model.
""")

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

col_upload, col_camera = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "📁 Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

with col_camera:
    camera_image = st.camera_input(
        "📷 Take Photo"
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None or camera_image is not None:

    if uploaded_file is not None:
        image_data = uploaded_file.read()
    else:
        image_data = camera_image.read()

    file_bytes = np.asarray(
        bytearray(image_data),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Preprocessing

    img_resized = cv2.resize(
        img,
        (128, 128)
    )

    img_resized = img_resized / 255.0

    img_resized = np.expand_dims(
        img_resized,
        axis=0
    )

    # Prediction

    prediction = model.predict(
        img_resized,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

    confidence = (
        np.max(prediction) * 100
    )

    disease = classes[
        predicted_class
    ]

    info = disease_info[disease]

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    report = f"""
Plant Disease Detection Report

Disease: {disease}

Confidence: {confidence:.2f}%

Scientific Name:
{info['scientific_name']}

Pathogen Type:
{info['pathogen_type']}

Symptoms:
{info['symptoms']}

Management:
{info['management']}

Prevention:
{info['prevention']}
"""

    # --------------------------------------------------
    # RESULT SECTION
    # --------------------------------------------------

    col_image, col_result = st.columns([2, 1])

    with col_image:

        st.image(
            cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            ),
            caption="🌿 Uploaded Leaf",
            use_container_width=True
        )

    with col_result:

        st.success(
            f"Detected Disease: {disease}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )

        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="disease_report.txt",
            mime="text/plain"
        )

    # --------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------

    st.subheader(
        "📊 Disease Probability Distribution"
    )

    prob_df = pd.DataFrame({
        "Disease": classes,
        "Probability": prediction[0] * 100
    })

    st.bar_chart(
        prob_df.set_index("Disease"),
        use_container_width=True
    )

    # --------------------------------------------------
    # DISEASE INFORMATION
    # --------------------------------------------------

    st.subheader(
        "📖 Disease Information"
    )

    with st.expander(
        "Scientific Name",
        expanded=True
    ):
        st.write(
            info["scientific_name"]
        )

    with st.expander(
        "Pathogen Type"
    ):
        st.write(
            info["pathogen_type"]
        )

    with st.expander(
        "Symptoms"
    ):
        st.write(
            info["symptoms"]
        )

    with st.expander(
        "Management"
    ):
        st.write(
            info["management"]
        )

    with st.expander(
        "Prevention"
    ):
        st.write(
            info["prevention"]
        )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.markdown("---")

st.header("📈 Model Performance")

with st.expander("View Training Results"):

    st.image(
        "accuracy_plot.png",
        caption="Accuracy vs Epoch",
        use_container_width=True
    )

    st.image(
        "loss_plot.png",
        caption="Loss vs Epoch",
        use_container_width=True
    )

    st.image(
        "confusion_matrix.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

st.markdown("---")

st.header("📖 About Project")

st.markdown("""
### Dataset
PlantVillage Dataset

### Classes
- Potato Early Blight
- Potato Healthy
- Tomato Early Blight
- Tomato Late Blight
- Tomato Healthy

### Model
Custom CNN using TensorFlow and Keras

### Validation Accuracy
94.69%

### Technologies Used
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Streamlit
""")

st.markdown("---")

st.caption(
    "Developed by D Haritha | AI/ML Internship Project | 2026"
)
