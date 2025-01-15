import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained model
MODEL_PATH = "sapling_survival_model.h5"
model = load_model(MODEL_PATH)


# Preprocessing function
def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Model's input size
    return image / 255.0  # Normalize


# Calculate survival percentage and visualize
def calculate_survival(images_dir):
    """Calculate survival percentage and casualty percentage."""
    total = 0
    alive_count = 0
    for file_name in os.listdir(images_dir):
        file_path = os.path.join(images_dir, file_name)
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                image = preprocess_image(file_path)
                prediction = model.predict(np.expand_dims(image, axis=0))
                status = "Alive" if prediction[0][0] > 0.5 else "Dead"
                total += 1
                if status == "Alive":
                    alive_count += 1
            except Exception as e:
                st.warning(f"Error processing file {file_path}: {e}")
    survival_percentage = (alive_count / total) * 100 if total > 0 else 0
    casualty_percentage = 100 - survival_percentage
    return survival_percentage, casualty_percentage


# Streamlit app
def main():
    st.title("Sapling Survival Analysis")
    st.sidebar.title("Upload Raw Data")

    # Upload Raw Data Directory
    raw_data_dir = st.sidebar.text_input(
        "Enter the path to the Raw Data directory (e.g., ./Raw_Data):"
    )

    if raw_data_dir and os.path.isdir(raw_data_dir):
        st.write(f"Raw Data Directory: {raw_data_dir}")

        # Calculate survival rate
        if st.button("Analyze Survival"):
            with st.spinner("Analyzing..."):
                survival_rate, casualty_rate = calculate_survival(raw_data_dir)
                st.success("Analysis Completed!")
                st.metric("Survival Rate (%)", f"{survival_rate:.2f}")
                st.metric("Casualty Rate (%)", f"{casualty_rate:.2f}")

                # Bar chart visualization
                labels = ['Survived', 'Casualties']
                values = [survival_rate, casualty_rate]
                colors = ['green', 'red']

                fig, ax = plt.subplots()
                ax.bar(labels, values, color=colors, alpha=0.8)
                ax.set_title('Sapling Survival Rate')
                ax.set_ylabel('Percentage')
                for i, v in enumerate(values):
                    ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)
                st.pyplot(fig)
    else:
        st.warning("Please provide a valid Raw Data directory path.")


if __name__ == "__main__":
    main()
