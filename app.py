import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from pathlib import Path


st.set_page_config(
    page_title="Image Analysis",
    page_icon="🖼️",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

.block-container {
    max-width: 1000px;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path(__file__).parent / "models" / "cats_dogs_classifier.keras"

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH)


uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    rgb_array = np.array(image)

    bgr_image = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2BGR
    )
    gray_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2GRAY
    )
    st.image(
        rgb_array,
        caption="Original Image",
        width=450
    )

    st.subheader("Image Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("Width", image.width)
    c2.metric("Height", image.height)
    c3.metric("Channels", rgb_array.shape[2])

    processing_tab, edges_tab, stats_tab, ai_tab = st.tabs([
        "Processing",
        "Edges",
        "Statistics",
        "Classification"
    ])

    with processing_tab:

        # Grayscale
        st.subheader("Grayscale")

        st.image(
            gray_image,
            caption="Grayscale Image",
            use_container_width=True
        )


        # Gaussian Blur
        st.subheader("Gaussian Blur")

        kernel = st.selectbox(
            "Blur intensity",
            [3, 5, 7, 9, 15],
            index=1
        )

        blurred = cv2.GaussianBlur(
            bgr_image,
            (kernel, kernel),
            0
        )

        st.image(
            cv2.cvtColor(
                blurred,
                cv2.COLOR_BGR2RGB
            ),
            caption="Blurred Image",
            use_container_width=True
        )
        st.subheader("Thresholding")

        threshold = st.number_input(
            "Threshold value",
            min_value=0,
            max_value=255,
            value=127
        )

        _, binary = cv2.threshold(
            gray_image,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        st.image(
            binary,
            caption="Binary Image",
            use_container_width=True
        )

    with edges_tab:
        st.subheader("Canny Edge Detection")

        lower = st.number_input(
            "Lower threshold",
            min_value=0,
            max_value=255,
            value=100
        )

        upper = st.number_input(
            "Upper threshold",
            min_value=0,
            max_value=255,
            value=200
        )

        if lower < upper:

            edges = cv2.Canny(
                gray_image,
                lower,
                upper
            )

            st.image(
                edges,
                caption="Detected Edges",
                use_container_width=True
            )

        else:
            st.warning(
                "Upper threshold must be greater than lower threshold."
            )

        contours, _ = cv2.findContours(
            binary.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contour_image = bgr_image.copy()

        cv2.drawContours(
            contour_image,
            contours,
            -1,
            (0, 255, 0),
            2
        )

        st.subheader("Contours")

        st.image(
            cv2.cvtColor(
                contour_image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        st.metric(
            "Contours Detected",
            len(contours)
        )
        min_area = st.number_input(
            "Minimum contour area",
            min_value=0,
            value=1000,
            step=500
        )

        box_image = bgr_image.copy()
        count = 0

        for contour in contours:

            if cv2.contourArea(contour) > min_area:

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    box_image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                count += 1

        st.subheader("Bounding Boxes")

        st.image(
            cv2.cvtColor(
                box_image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        st.metric(
            "Contours Above Minimum Area",
            count
        )


    with stats_tab:

        # Average brightness
        brightness = np.mean(gray_image)

        st.metric(
            "Average Brightness",
            f"{brightness:.2f} / 255"
        )


        # Contrast using standard deviation
        contrast = np.std(gray_image)

        st.metric(
            "Contrast",
            f"{contrast:.2f}"
        )


        # Grayscale histogram
        histogram = cv2.calcHist(
            [gray_image],
            [0],
            None,
            [256],
            [0, 256]
        ).flatten()

        st.subheader("Grayscale Histogram")

        st.line_chart(histogram)


    with ai_tab:

        st.subheader("Cats vs Dogs Classification")

        if MODEL_PATH.exists():

            try:

                model = load_cnn_model()

                resized_image = image.resize((128, 128))

                input_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                input_array = np.expand_dims(
                    input_array,
                    axis=0
                )

                prediction = float(
                    model.predict(
                        input_array,
                        verbose=0
                    )[0][0]
                )

                if prediction >= 0.5:
                    label = "Dog 🐶"
                    confidence = prediction * 100

                else:
                    label = "Cat 🐱"
                    confidence = (1 - prediction) * 100


                st.image(
                    rgb_array,
                    caption="Image given to the CNN",
                    use_container_width=True
                )

                st.success(
                    f"Prediction: {label}"
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            except Exception as e:

                st.error(
                    f"Could not load or run the CNN model: {e}"
                )

        else:

            st.error(
                "CNN model not found. Place "
                "'cats_dogs_classifier.keras' inside the models folder."
            )


else:

    st.write("")
