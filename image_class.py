import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Caching the model loading so it doesn't reload on every run.
@st.cache_resource
def load_model():
    # Load MobileNetV2 with pre-trained ImageNet weights.
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

model = load_model()

st.title("Image Classification Tool")
st.write("""
Upload an image and let MobileNetV2 classify it. The app displays the top predicted classes with confidence scores.
You can also view some intermediate feature maps for extra insight.
""")

# File uploader accepts common image file types.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image with PIL.
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("### Classifying...")
    # MobileNetV2 expects 224x224 images
    img_resized = image.resize((224, 224))
    x = np.array(img_resized)

    # If the image has an alpha channel, drop it.
    if x.shape[-1] == 4:
        x = x[..., :3]

    # Expand dims to add the batch dimension.
    x = np.expand_dims(x, axis=0)
    # Preprocess the image using the appropriate function.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Make predictions.
    preds = model.predict(x)
    # Decode the predictions into class names and confidence scores.
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

    st.write("### Predictions:")
    for i, pred in enumerate(decoded_preds):
        st.write(f"**{pred[1]}**: {pred[2]*100:.2f}%")

    # Optionally, display feature maps.
    if st.checkbox("Show Feature Maps"):
        st.write("### Feature Maps from an Intermediate Layer")
        try:
            # Create a new model that outputs the activations from an intermediate layer.
            # 'out_relu' is the last activation layer before the global pooling.
            intermediate_layer_model = tf.keras.Model(
                inputs=model.input, 
                outputs=model.get_layer("out_relu").output
            )
        except ValueError:
            st.error("Could not find the feature map layer in the model.")
        else:
            feature_maps = intermediate_layer_model.predict(x)
            # feature_maps shape is (1, height, width, channels)
            # We'll display the first 16 feature maps.
            num_features = feature_maps.shape[-1]
            num_display = min(16, num_features)

            # Create a grid for displaying the feature maps.
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            for i, ax in enumerate(axes.flat):
                if i < num_display:
                    # Extract the i-th feature map.
                    fm = feature_maps[0, :, :, i]
                    ax.imshow(fm, cmap='viridis')
                    ax.set_title(f"Filter {i}")
                    ax.axis("off")
                else:
                    ax.axis("off")
            st.pyplot(fig)
