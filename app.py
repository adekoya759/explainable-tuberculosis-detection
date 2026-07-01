import streamlit as st
from PIL import Image
from model_helper import (
    predict,
    generate_heatmap,
    class_names
)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="TB Detection using DenseNet121",
    page_icon="🫁",
    layout="wide"
)

# ======================================================
# TITLE
# ======================================================
st.title("🫁 Tuberculosis Detection using DenseNet121")
st.write(
    """
    Upload a chest X-ray image to detect whether the patient
    is **Tuberculosis positive** or **Normal (Control)**.
    The system also provides:
    - Prediction confidence
    - Class probabilities
    - Grad-CAM++ explainability heatmap
    """
)

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# PREDICTION
# ======================================================
if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, width=500)



    with st.spinner("Analyzing image..."):

        prediction, confidence, probabilities = predict(image)

        heatmap = generate_heatmap(image)

    # ==================================================
    # RESULTS
    # ==================================================
    st.subheader("Prediction")

    if prediction == "Tuberculosis":
        st.error(
            f"Prediction: {prediction}"
        )
    else:
        st.success(
            f"Prediction: {prediction}"
        )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # ==================================================
    # PROBABILITIES
    # ==================================================
    st.subheader("Class Probabilities")

    for i, prob in enumerate(probabilities):

        st.progress(float(prob))

        st.write(
            f"{class_names[i]}: "
            f"{prob.item()*100:.2f}%"
        )

    # ==================================================
    # HEATMAP
    # ==================================================

    with col2:
        st.subheader("Grad-CAM++ Explainability")
        st.image(heatmap, width=500)


# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption(
    "Developed using DenseNet121 and Grad-CAM++ "
    "for explainable tuberculosis detection."
)