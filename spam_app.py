import matplotlib.pyplot as plt
import streamlit as st
import pickle

# Set the page title and layout
st.set_page_config(page_title="Spam Email Detector", layout="centered")

# Load the trained pipeline (TF-IDF + Naive Bayes)
model = pickle.load(open('spam_model.pkl', 'rb'))

# Streamlit App Title
st.title("📧 Spam Email Classifier")
st.markdown("Detect whether a given email is **Spam** or **Not Spam** using a trained ML model.")

# Text input
input_email = st.text_area("✉️ Enter email content below:")

import matplotlib.pyplot as plt

if st.button("Predict"):
    if input_email.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        result = model.predict([input_email])[0]
        proba = model.predict_proba([input_email])[0][1]  # Spam probability

        confidence = proba if result == 1 else 1 - proba
        label = "🚫 This is **Spam**." if result == 1 else "✅ This is **Not Spam**."

        # Display prediction label
        st.markdown(f"### {label}")
        st.info(f"🔍 Confidence: **{confidence * 100:.2f}%**")

        # Plot confidence bar
        st.markdown("#### 🔢 Visual Confidence")
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh([0], [confidence * 100], color="skyblue")
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Spam Detection Confidence")
        st.pyplot(fig)
