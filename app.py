import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================================================
# AI-Based PPE Compliance Detection System
# Developed by Adithya S S
# =========================================================


st.set_page_config(
    page_title="PPE Compliance Detection System",
    page_icon=":material/shield:",
    layout="wide"
)

with st.container(horizontal_alignment="center"):
    st.title(
        "PPE Compliance Detection System",
        icon=":material/shield:",
        text_alignment="center",
    )
    st.caption("AI-Based PPE Compliance Verification")
    st.badge(
        "Developed by Adithya S S",
        icon=":material/science:",
        color="blue",
    )


# ---------------------------------------------------------
# MODEL DOWNLOAD
# ---------------------------------------------------------

MODEL_FOLDER = "models"
MODEL_PATH = os.path.join(MODEL_FOLDER, "sh17_yolov8m.pt")
TRAINED_MODEL_PATH = os.path.join(
    "runs",
    "detect",
    "ppe-yolov8m",
    "weights",
    "best.pt"
)

@st.cache_resource
def load_model():
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    model_path = TRAINED_MODEL_PATH if os.path.exists(TRAINED_MODEL_PATH) else MODEL_PATH
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the PPE AI model.")
    st.write("Error:", e)
    st.stop()


# ---------------------------------------------------------
# PPE DEFINITIONS
# ---------------------------------------------------------

PPE_ITEMS = {
    "face-mask": "Face Mask",
    "glasses": "Eye protection (glasses)",
    "gloves": "Gloves",
    "shoes": "Safety Shoes",
    "safety-suit": "Safety Suit",
    "medical-suit": "Protective Suit"
}

with st.sidebar:
    st.subheader("Settings", icon=":material/tune:")
    st.caption("Adjust how the image detector runs.")
    with st.container(border=True):
        confidence_threshold = st.slider(
            "Minimum confidence",
            min_value=0.10,
            max_value=0.70,
            value=0.25,
            step=0.05,
            help="Lower values can reveal smaller or less certain PPE, but may add false detections."
        )
        image_size = st.select_slider(
            "Inference resolution",
            options=[640, 960, 1280],
            value=1280,
            help="Higher resolution can help with small PPE details, at the cost of slower inference."
        )
        use_augmentation = st.toggle(
            "Use test-time augmentation",
            value=True,
            help="Runs augmented inference to improve detection of difficult objects; takes longer."
        )
    st.caption(
        "Lower confidence may find more PPE but can add false positives. "
        "Higher confidence is stricter but may miss PPE."
    )
    with st.expander("About this project", icon=":material/info:"):
        st.caption("Developed by Adithya S S")


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

st.header("Check PPE", icon=":material/image_search:")
st.caption("Upload a clear image, then review the AI detections with a person.")
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload an image for PPE review",
        type=["jpg", "jpeg", "png"],
        help="For clearer results, use a well-lit image where the person and PPE are visible.",
    )


# ---------------------------------------------------------
# IMAGE PROCESSING
# ---------------------------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Original image", icon=":material/photo:")
            st.image(
                image,
                width="stretch",
                alt="Uploaded image for PPE detection"
            )


    # -----------------------------------------------------
    # RUN AI
    # -----------------------------------------------------

    with st.spinner("Running PPE detection..."):

        results = model(
            image,
            conf=confidence_threshold,
            imgsz=image_size,
            augment=use_augmentation,
            max_det=300
        )

    result = results[0]


    # -----------------------------------------------------
    # DETECTED ITEMS
    # -----------------------------------------------------

    detected_confidence = {}
    person_count = 0

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        confidence = float(box.conf.item())

        if class_name == "person":
            person_count += 1
        detected_confidence[class_name] = max(
            confidence,
            detected_confidence.get(class_name, 0.0)
        )

    detected_items = set(detected_confidence)


    # -----------------------------------------------------
    # DISPLAY AI RESULT
    # -----------------------------------------------------

    annotated_image = result.plot()

    with col2:
        with st.container(border=True):
            st.subheader("AI detections", icon=":material/center_focus_strong:")
            st.image(
                annotated_image,
                channels="BGR",
                width="stretch",
                alt="Image annotated with detected PPE objects"
            )


    st.header("PPE checklist", icon=":material/checklist:")
    st.caption(
        f"{person_count} person detection(s) · confidence threshold "
        f"{confidence_threshold:.0%}"
    )


    # -----------------------------------------------------
    # PPE STATUS
    # -----------------------------------------------------

    status_columns = st.columns(3, gap="medium")
    for index, (class_name, display_name) in enumerate(PPE_ITEMS.items()):
        with status_columns[index % len(status_columns)]:
            with st.container(border=True):
                st.markdown(f"**{display_name}**")
                if class_name in detected_items:
                    st.badge(
                        f"Detected · {detected_confidence[class_name]:.0%}",
                        icon=":material/check_circle:",
                        color="green",
                    )
                else:
                    st.badge(
                        "Not detected",
                        icon=":material/help:",
                        color="orange",
                    )


    # -----------------------------------------------------
    # ACCESS LOGIC
    # -----------------------------------------------------

    # Required items for this exhibition prototype.
    #
    # The model does not have a dedicated head-cover class,
    # so head cover is NOT claimed as detected here.

    required_items = [
        "face-mask",
        "glasses",
        "gloves",
        "shoes"
    ]


    missing_items = []

    for item in required_items:

        if item not in detected_items:

            missing_items.append(
                PPE_ITEMS[item]
            )


    st.header("Review outcome", icon=":material/fact_check:")


    # -----------------------------------------------------
    # ACCESS DECISION
    # -----------------------------------------------------

    if person_count == 0:
        st.warning(
            "Unable to make a PPE decision: no person was detected. "
            "Try a clearer, closer, well-lit image.",
            icon=":material/person_search:",
        )

    elif len(missing_items) == 0:
        st.warning(
            "Manual verification required: all required PPE classes were "
            "detected, but this model cannot confirm they are correctly worn "
            "or certified. Do not use this result to grant access automatically.",
            icon=":material/visibility:",
        )

    else:
        with st.container(border=True):
            st.warning(
                "Some required PPE classes were not detected. This does not "
                "prove they are absent; verify the image manually.",
                icon=":material/warning:",
            )
            for item in missing_items:
                st.badge(
                    f"{item} · verify",
                    icon=":material/search:",
                    color="orange",
                )


    # -----------------------------------------------------
    # DETECTION DETAILS
    # -----------------------------------------------------

    with st.expander("Detection details", icon=":material/analytics:"):
        if detected_items:
            st.caption("Detected classes and their highest confidence scores.")
            detail_columns = st.columns(4, gap="small")
            for index, item in enumerate(sorted(detected_items)):
                with detail_columns[index % len(detail_columns)]:
                    st.metric(item.replace("-", " ").title(), f"{detected_confidence[item]:.0%}")
        else:
            st.info("No supported PPE items were detected.")


# ---------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------

else:

    with st.container(border=True):
        st.subheader("How the demonstration works", icon=":material/route:")
        step_columns = st.columns(3, gap="medium")
        steps = [
            ("Upload", "Choose a clear image of a person."),
            ("Detect", "The pretrained model identifies PPE-related objects."),
            ("Review", "Check detections and confidence with a person."),
        ]
        for column, (title, description) in zip(step_columns, steps):
            with column:
                st.markdown(f"**{title}**")
                st.caption(description)

    with st.container(border=True):
        st.info(
            "This is an exhibition prototype for human review. Detections do "
            "not confirm that PPE is correctly worn or certified, and the app "
            "does not control a physical door.",
            icon=":material/health_and_safety:",
        )

st.caption("© Adithya S S. All rights reserved.")