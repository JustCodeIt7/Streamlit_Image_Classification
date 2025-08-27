import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
import io
import json
from typing import List, Tuple, Optional

# Set page configuration (optional, but good practice)
st.set_page_config(
    page_title="PyTorch Image Recognition App",
    page_icon="🖼️",
    layout="centered",
)

# --- Model and Label Loading (Cached) ---


@st.cache_resource  # Use cache_resource for non-data objects like models
def load_model(model_name: str = "resnet18"):
    """Loads a pre-trained PyTorch model."""
    st.write(f"Loading {model_name} model...")
    if model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )  # Use V2 for potentially better accuracy
        model = torchvision.models.resnet50(weights=weights)
    elif model_name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=weights)
    else:
        st.error(f"Model {model_name} not supported.")
        return None, None

    model.eval()  # Set model to evaluation mode (important!)
    preprocess = weights.transforms()  # Get the recommended transforms for the model
    st.write("Model loaded successfully!")
    return model, preprocess


@st.cache_data  # Use cache_data for data like labels
def load_imagenet_labels() -> Optional[List[str]]:
    """Loads ImageNet class labels."""
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        response = requests.get(LABELS_URL)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        labels = response.json()
        return labels
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching ImageNet labels: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding ImageNet labels JSON.")
        return None


# --- Image Processing and Prediction ---


def predict(
    model, preprocess, image: Image.Image, top_k: int = 5
) -> Optional[List[Tuple[str, float]]]:
    """Processes an image and returns top K predictions."""
    if image.mode != "RGB":
        image = image.convert("RGB")  # Ensure image is in RGB format

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # Create a mini-batch as expected by the model

    # Move tensor to GPU if available (optional, but faster)
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():  # Turn off gradient calculation for inference
        output = model(input_batch)

    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Load labels
    imagenet_labels = load_imagenet_labels()
    if not imagenet_labels:
        st.error("Cannot make predictions without ImageNet labels.")
        return None

    # Get top K predictions
    top_prob, top_indices = torch.topk(probabilities, top_k)

    # Map indices to labels and probabilities
    predictions = [
        (imagenet_labels[idx], prob.item()) for idx, prob in zip(top_indices, top_prob)
    ]

    return predictions


# --- Streamlit App UI ---

st.title("🖼️ PyTorch Image Recognition")
st.write(
    "Upload an image, and this app will classify it using a pre-trained PyTorch model (ResNet or EfficientNet)."
)

# --- Sidebar for Options ---
st.sidebar.header("⚙️ Options")
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    ("resnet18", "resnet50", "efficientnet_b0"),
    index=0,  # Default to resnet18
)
top_k_slider = st.sidebar.slider(
    "Number of Predictions to Show:", min_value=1, max_value=10, value=5
)

# Load selected model and its preprocessing steps
model, preprocess = load_model(model_choice)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None and preprocess is not None:
    # Read the image file
    try:
        image = Image.open(uploaded_file)

        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform prediction
        with st.spinner("🧠 Classifying..."):
            predictions = predict(model, preprocess, image, top_k=top_k_slider)

        # Display predictions
        with col2:
            st.subheader(f"Top {top_k_slider} Predictions:")
            if predictions:
                for label, probability in predictions:
                    st.write(f"- {label}: {probability:.2%}")
            else:
                st.error("Prediction failed.")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.error("Please try uploading a valid image file (JPG, JPEG, PNG).")

elif uploaded_file is None:
    st.info("Please upload an image file.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with [Streamlit](https://streamlit.io) & [PyTorch](https://pytorch.org)"
)
