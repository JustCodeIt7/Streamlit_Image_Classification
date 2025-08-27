import streamlit as st
import torch
from torchvision import models
from PIL import Image
import pandas as pd
from torch.nn import functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

# The logical order for a tutorial would be:
# 1. setup_ui: Create the user interface first. It's visual and easy to grasp.
# 2. load_model: Explain how to load the pre-trained model, the "brain" of the app.
# 3. predict: Show the core logic of making a prediction on an image.
# 4. display_predictions: Explain how to present the results to the user.
# 5. generate_gradcam: Introduce the advanced concept and deep learning magic of Grad-CAM.
# 6. display_gradcam: Show how to visualize the Grad-CAM output.
# 7. main: Tie everything together in the main application flow.

# --- Core Application Logic Methods ---


@st.cache_resource
def load_model(model_name):
    """
    Loads a pre-trained model from torchvision, its weights, preprocessing transforms,
    and category labels. The st.cache_resource decorator ensures the model is loaded
    only once.
    """
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


def predict(model, preprocess, image, categories, top_k):
    """
    Takes a model and an image, preprocesses the image, and returns the top k
    predictions as a pandas DataFrame.
    """
    # Ensure image is in RGB format
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Preprocess the image and add a batch dimension
    img_tensor = preprocess(image).unsqueeze(0)

    # Get model predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)

    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame(
        {
            "Class": [categories[i] for i in top_indices],
            "Probability": top_probs.numpy() * 100,
        }
    )

    return predictions_df


def generate_gradcam(model, img_tensor, target_class=None):
    """
    Generates a Grad-CAM heatmap for a given model and image tensor.
    """
    # Store activations and gradients
    activations = {}
    gradients = {}

    # Define hook functions
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()

        return hook

    # Register hooks to the target layer
    # For ResNet, the final convolutional block is 'layer4'
    # For EfficientNet, it's 'features[-1]'
    target_layer = model.layer4 if hasattr(model, "layer4") else model.features[-1]
    handle_forward = target_layer.register_forward_hook(save_activation("target_layer"))
    handle_backward = target_layer.register_backward_hook(save_gradient("target_layer"))

    # Forward pass
    outputs = model(img_tensor)
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    outputs[0, target_class].backward()

    # Get activations and gradients
    acts = activations["target_layer"]
    grads = gradients["target_layer"]

    # Compute weights (global average pooling on gradients)
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)

    # Generate heatmap
    cam = torch.sum(weights * acts, dim=1).squeeze(0)
    cam = F.relu(cam)  # Apply ReLU

    # Normalize heatmap
    cam = cam / (torch.max(cam) + 1e-8)

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return cam.cpu().numpy(), categories[target_class]


# --- UI and Display Methods ---


def setup_ui():
    """
    Sets up the Streamlit page configuration and sidebar widgets.
    Returns the user's choices.
    """
    st.set_page_config(
        page_title="Image Classification with PyTorch",
        page_icon="🖼️",
        layout="centered",
    )
    st.title("🖼️ Image Classification with PyTorch")

    with st.expander("About this app"):
        st.markdown(
            """
        This app uses pre-trained PyTorch models to classify images.
        **How it works:**
        1. Select a model from the sidebar.
        2. Upload an image.
        3. The app displays the top predictions and their probabilities.
        4. For ResNet models, you can also view a Grad-CAM heatmap to see where the model "looks".
        """
        )

    st.sidebar.header("⚙️ Options")
    model_choice = st.sidebar.selectbox(
        "Choose a Model:",
        ("resnet18", "resnet50", "efficientnet_b0"),
        index=1,  # Default to resnet50
    )
    top_k_slider = st.sidebar.slider(
        "Number of Predictions to Show:", min_value=1, max_value=10, value=5
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    return model_choice, top_k_slider, uploaded_file


def display_predictions(predictions_df):
    """
    Displays the classification results and a bar chart of probabilities.
    """
    st.header("Classification Results")
    for _, row in predictions_df.iterrows():
        st.write(f"**{row['Class']}**: {row['Probability']:.2f}%")

    st.subheader("Probability Distribution")
    st.bar_chart(predictions_df.set_index("Class"))


def display_gradcam(model, preprocess, image, categories):
    """
    Handles the Grad-CAM visualization display logic.
    """
    st.header("Grad-CAM Visualization")

    with st.spinner("Generating heatmap..."):
        # Preprocess image for Grad-CAM
        if image.mode == "RGBA":
            image = image.convert("RGB")
        img_tensor = preprocess(image).unsqueeze(0)

        # Generate and get heatmap
        heatmap, pred_class_name = generate_gradcam(model, img_tensor)

        # Prepare images for plotting
        img_np = np.array(image.resize((224, 224)))
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )

        # Superimpose heatmap on the original image
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        # Plotting
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(img_np)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2.imshow(heatmap, cmap="jet")
        ax2.set_title("Activation Heatmap")
        ax2.axis("off")

        ax3.imshow(superimposed_img)
        ax3.set_title(f"Attention for: {pred_class_name}")
        ax3.axis("off")

        st.pyplot(fig)

        st.info(
            f"""
            **Grad-CAM Explanation:** The heatmap shows where the model is focusing to predict 
            **"{pred_class_name}"**. Bright regions (red/yellow) are the most influential in this decision.
            """
        )


# --- Main Application Entry Point ---


def main():
    """
    Main function to run the Streamlit application.
    """
    model_choice, top_k, uploaded_file = setup_ui()

    try:
        model, preprocess, categories = load_model(model_choice)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)

        with col1:
            st.header("Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            with st.spinner("Classifying..."):
                predictions_df = predict(model, preprocess, image, categories, top_k)
                display_predictions(predictions_df)

        # Grad-CAM option appears below the main results
        show_gradcam = st.checkbox("Show Activation Heatmap (Grad-CAM)")
        if show_gradcam:
            if "resnet" in model_choice:
                display_gradcam(model, preprocess, image, categories)
            else:
                st.warning(
                    "Grad-CAM is currently optimized for ResNet models in this app."
                )


if __name__ == "__main__":
    main()
