import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------- Define the model architecture ----------
def build_model():
    img_size = 224
    base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------- Load model and weights ----------
model = build_model()
model.load_weights('model_epoch_99_weights.h5')  # update with correct path

# ---------- Streamlit app ----------
st.title("🕵️ Deepfake Detector")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    label = "Fake" if prediction >= 0.5 else "Real"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"### Prediction: {label}")
    st.markdown(f"### Confidence: {confidence * 100:.2f}%")
