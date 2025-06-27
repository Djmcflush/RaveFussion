import streamlit as st
import torch
from riffusion import riffusion_pipeline, spectrogram_params, audio_splitter
from PIL import Image
import numpy as np

st.set_page_config(page_title="RaveFussion", page_icon="🎵", layout="wide")

st.title("🎵 RaveFussion - AI Music Generator")

st.sidebar.header("Settings")

# Input text for music generation
text_prompt = st.text_input("Enter a description of the music you want to generate", "epic electronic dance music with heavy bass")

# Music parameters
col1, col2 = st.columns(2)
with col1:
    seed = st.number_input("Seed", value=42)
    num_inference_steps = st.slider("Number of inference steps", min_value=10, max_value=100, value=50)

with col2:
    guidance_scale = st.slider("Guidance scale", min_value=1.0, max_value=20.0, value=7.0)
    duration = st.slider("Duration (seconds)", min_value=1, max_value=30, value=5)

# Generate button
if st.button("Generate Music"):
    with st.spinner("Generating your music..."):
        try:
            # Initialize the pipeline
            pipeline = riffusion_pipeline.RiffusionPipeline.from_pretrained(
                "./riffusion-model-v1",
                local_files_only=True
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # Set up parameters
            params = spectrogram_params.SpectrogramParams()
            
            # Generate the audio
            image = pipeline(
                prompt=text_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            ).images[0]

            # Convert spectrogram to audio
            audio = audio_splitter.audio_from_spectrogram(image, params)
            
            # Display the spectrogram
            st.image(image, caption="Generated Spectrogram", use_column_width=True)
            
            # Play the audio
            st.audio(audio, sample_rate=params.sample_rate)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown("""
### How to use:
1. Enter a description of the music you want to generate
2. Adjust the generation parameters if desired
3. Click 'Generate Music' and wait for the result
4. Listen to your generated music and view the spectrogram
""")
