import streamlit as st
import torch
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import functional as F
import numpy as np
import cv2

# Set page configuration
st.set_page_config(
    page_title="Image Classification with PyTorch",
    page_icon="🖼️",
    layout="centered",
)


# Caching the model loading so it doesn't reload on every run
@st.cache_resource
def load_model(model_name):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()  # Set the model to evaluation mode
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories


# Function to get feature maps from intermediate layers (kept for potential future use, but not currently called)
def get_activation_maps(img_tensor, model):
    # Create a hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations["layer4"] = output

    # Register the hook on the last layer (layer4 in ResNet)
    if hasattr(model, "layer4"):
        model.layer4.register_forward_hook(hook_fn)
    else:
        # For EfficientNet, use the last feature layer
        model.features[-1].register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)

    return activations


# Title and description
st.title("🖼️ Image Classification with PyTorch")
st.markdown(
    """
This app uses pre-trained PyTorch models (ResNet18, ResNet50, or EfficientNet-B0) to classify images. 
Upload an image to see what the model thinks it is, along with class probabilities and feature maps (for ResNet models).
"""
)

# Sidebar for options
st.sidebar.header("⚙️ Options")
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    ("resnet18", "resnet50", "efficientnet_b0"),
    index=1,  # Default to resnet50
)
top_k_slider = st.sidebar.slider(
    "Number of Predictions to Show:", min_value=1, max_value=10, value=5
)

# Load model
try:
    model, preprocess, categories = load_model(model_choice)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Create columns for layout
col1, col2 = st.columns([1, 1])

# Image upload in the left column
with col1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image with PIL
        image = Image.open(uploaded_file)
        # Convert RGBA to RGB if necessary
        if image.mode == "RGBA":
            image = image.convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Model prediction in the right column
with col2:
    if uploaded_file is not None and model_loaded:
        st.header("Classification Results")

        with st.spinner("Classifying..."):
            # Preprocess the image (ensure RGB)
            if image.mode == "RGBA":
                rgb_image = image.convert("RGB")
            else:
                rgb_image = image
            img_tensor = preprocess(rgb_image).unsqueeze(0)  # Add batch dimension

            # Get predictions
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]

            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, top_k_slider)

            # Display predictions
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                class_name = categories[idx]
                confidence = prob.item() * 100
                st.write(f"**{i + 1}. {class_name}**: {confidence:.2f}%")

            # Create a bar chart for top predictions
            st.subheader("Probability Distribution")

            # Convert top class names and probabilities to a DataFrame
            df = pd.DataFrame(
                {"Probability": top_probs.numpy() * 100},
                index=[categories[idx] for idx in top_indices],
            )

            # Create the bar chart
            st.bar_chart(df)


def generate_gradcam(img_tensor, model, target_class=None):
    """Generate Grad-CAM visualization for the predicted class"""
    # Store activations and gradients
    activations = {}
    gradients = {}

    # Determine the hook layer based on model type
    if hasattr(model, "layer4"):
        hook_layer = model.layer4
    else:
        # For EfficientNet, use the last feature layer
        hook_layer = model.features[-1]

    # Function to get activations during forward pass
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output

        return hook

    # Function to get gradients during backward pass
    def save_gradient(name):
        def hook(grad):
            gradients[name] = grad

        return hook

    # Register hooks
    handle_forward = hook_layer.register_forward_hook(save_activation("last_layer"))

    # Forward pass
    outputs = model(img_tensor)

    # If no target class is specified, use the predicted class
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()

    # One-hot encode the target class
    target = torch.zeros_like(outputs)
    target[0, target_class] = 1

    # Clear existing gradients
    model.zero_grad()

    # Get activations
    layer_activation = activations["last_layer"]

    # Register hook for gradients
    layer_activation.register_hook(save_gradient("last_layer"))

    # Backward pass
    outputs.backward(target, retain_graph=True)

    # Get gradients
    gradients = gradients["last_layer"][0]

    # Global average pooling of gradients
    weights = torch.mean(gradients, dim=(1, 2), keepdim=True)

    # Weighted combination of activation maps
    cam = torch.sum(weights * layer_activation[0], dim=0)

    # Apply ReLU
    cam = torch.maximum(cam, torch.tensor(0.0))

    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize CAM to match the input image size
    cam = cam.detach().cpu().numpy()

    # Clean up
    handle_forward.remove()

    return cam


if uploaded_file is not None and model_loaded:
    # Only show Grad-CAM for ResNet models (as EfficientNet architecture differs)
    if "resnet" in model_choice and st.checkbox("Show Activation Heatmap (Grad-CAM)"):
        st.header("Grad-CAM Visualization")

        with st.spinner("Generating heatmap..."):
            # Preprocess the image (ensure RGB)
            if image.mode == "RGBA":
                rgb_image = image.convert("RGB")
            else:
                rgb_image = image
            img_tensor = preprocess(rgb_image).unsqueeze(0)

            # Get predictions first to determine the predicted class
            with torch.no_grad():
                outputs = model(img_tensor)

            pred_class = outputs.argmax(dim=1).item()
            pred_class_name = categories[pred_class]

            # Generate Grad-CAM for the predicted class
            cam = generate_gradcam(img_tensor, model, pred_class)

            # Create a figure with two subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            img_np = np.array(rgb_image.resize((224, 224)))
            ax1.imshow(img_np)
            ax1.set_title("Original Image")
            ax1.axis("off")

            # Heatmap
            ax2.imshow(cam, cmap="jet")
            ax2.set_title("Activation Heatmap")
            ax2.axis("off")

            # Overlay heatmap on original image
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = np.uint8(255 * cam_resized)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Convert RGB to BGR for OpenCV
            img_np = (
                img_np[:, :, ::-1].copy()
                if img_np.shape[-1] == 3
                else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            )

            # Superimpose the heatmap on original image
            superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
            superimposed_img = superimposed_img[:, :, ::-1]  # Convert back to RGB

            ax3.imshow(superimposed_img)
            ax3.set_title(f"Attention for: {pred_class_name}")
            ax3.axis("off")

            st.pyplot(fig)

            st.markdown(
                f"""
            **Grad-CAM Visualization Explanation:**

            This visualization shows where the model is focusing to make its prediction. 
            Bright areas in the heatmap (red/yellow) are regions that strongly influence 
            the classification decision for "{pred_class_name}".
            """
            )
    elif "efficientnet" in model_choice and st.checkbox(
        "Show Activation Heatmap (Grad-CAM)"
    ):
        st.info(
            "Grad-CAM visualization is currently supported only for ResNet models due to architectural differences."
        )

# App information
with st.expander("About this app"):
    st.markdown(
        """
    This app uses pre-trained PyTorch models (ResNet18, ResNet50, EfficientNet-B0) trained on ImageNet.

    **How it works:**
    1. Select a model from the sidebar
    2. Upload an image
    3. The image is preprocessed (resized, normalized)
    4. The model returns probabilities for 1,000 different classes
    5. The app displays the top predictions

    Grad-CAM visualization (available for ResNet models) shows the activations from the last convolutional layer, visualizing what patterns the model detects in your image.
    """
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with Streamlit [<sup>1</sup>](https://streamlit.io) & [PyTorch](https://pytorch.org)"
)
