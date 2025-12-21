import streamlit as st
import torch
import numpy as np
from src.preprocessing import preprocessing 
from src.modeling import CNNModel           
import os
from config.config import RESULT_DIR

# Map numbers to names (This was missing!)
INT_TO_EMOTION = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}

EMOTION_DETAILS = {
    'neutral': '😐 Neutral',
    'calm': '😌 Calm',
    'happy': '😊 Happy',
    'sad': '😢 Sad',
    'angry': '😡 Angry',
    'fearful': '😨 Fearful',
    'disgust': '🤢 Disgust',
    'surprised': '😲 Surprised'
}

@st.cache_resource
def load_model():
    model = CNNModel(num_classes=8)
    model.load_state_dict(torch.load(RESULT_DIR/"emotion_cnn.h5", map_location="cpu"))
    model.eval()
    return model

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio clip to see the AI's prediction.")

# FIXED: Only one uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "flac", "mp3", "m4a"])

if uploaded_file is not None:
    # FIXED: Use a consistent temp name or track the extension
    file_extension = uploaded_file.name.split('.')[-1]
    temp_filename = f"temp_audio.{file_extension}"

    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Play audio back to the user
    st.audio(temp_filename)

    if st.button("Analyze Emotion"):
        try:
            with st.spinner("Processing..."):
                # 1. Run PREPROCESSING on the correct filename
                features = preprocessing(temp_filename) 
                
                # 2. Convert to Tensor
                feature_tensor = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)

                # 3. Predict
                model = load_model()
                with torch.no_grad():
                    output = model(feature_tensor)
                    prediction_idx = torch.argmax(output, dim=1).item()
                    
                    # FIXED: Map index -> name string -> emoji string
                    emotion_name = INT_TO_EMOTION[prediction_idx]
                    display_text = EMOTION_DETAILS[emotion_name]

                # 4. Show Result
                st.metric(label="Detected Emotion", value=display_text)
                st.balloons()
        
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        
        finally:
            # Cleanup: Remove the file after processing
            if os.path.exists(temp_filename):
                os.remove(temp_filename)