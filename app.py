#!/usr/bin/env python

import sys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify, request, abort
    import requests
    logger.debug("Basic imports successful")
    
    logger.debug("Attempting to import riffusion_layers")
    from riffusion_layers import TextLayer
    logger.debug("riffusion_layers imported successfully")
    
    logger.debug("Attempting to import rave")
    from rave import Raven
    logger.debug("rave imported successfully")
    
    logger.debug("Attempting to import clip")
    import clip
    logger.debug("clip imported successfully")
    
    logger.debug("Attempting to import diffusers")
    from diffusers import DiffusionPipeline
    logger.debug("diffusers imported successfully")
    
    import os
except Exception as e:
    logger.error(f"Error during imports: {str(e)}")
    raise


# Initialize models as None first
model = None
preprocess = None
diffusion_pipeline = None
riffusion_model = None
rave_model = None

def init_models():
    global model, preprocess, diffusion_pipeline, rave_model
    try:
        print("Initializing CLIP model...")
        model, preprocess = clip.load("ViT-L/14", device='cpu')
        print("CLIP model loaded successfully")

        print("Initializing Diffusion Pipeline...")
        diffusion_pipeline = DiffusionPipeline.from_pretrained(
            "./models/riffusion",
            local_files_only=True
        )
        print("Diffusion Pipeline initialized successfully")

        print("Initializing Rave model...")
        rave_model = Raven()
        print("Rave model initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        raise


app = Flask(__name__, static_folder="./reactapp/build", static_url_path="/")
app.logger.setLevel('DEBUG')


# serve react app
# https://blog.miguelgrinberg.com/post/how-to-deploy-a-react--flask-project
@app.route("/")
def index():
    return app.send_static_file("public/index.html")


# Define the microservice that instantiates the global model
@app.route("/init_model", methods=["POST"])
def init_model():
    """Create Riffusion Model with text embeddings"""
    global riffusion_model
    data = request.get_json()

    # Check if text_labels is present in the request JSON
    if "text_labels" not in data:
        return jsonify({"error": "text_labels missing from request"}), 400
    text_labels = data["text_labels"]
    riffusion_model = TextLayer(
        model, preprocess, diffusion_pipeline, text_labels=text_labels
    )
    riffusion_model.create_inital_audio()
    coordinates = riffusion_model.embeddings_coordinates_pca
    # Return a success response
    return jsonify({"status": "success", "coordinates": coordinates})


# Define the riffusion microservice
@app.route("/text_to_audio", methods=["POST"])
def text_to_audio():
    """Generate audio based on text"""
    data = request.get_json()

    # Check if text_labels is present in the request JSON
    if "cursor_coordinates" not in data:
        return jsonify({"error": "cursor_coordinate pair missing from request"}), 400
    coordinate_pair = data["cursor_coordinates"]
    save_path = data.get("save_path", None)
    file_path = riffusion_model.save_new_embedding(coordinate_pair, save_path)
    return jsonify({"result": file_path})


# Define the rave microservice
@app.route("/audio_to_audio", methods=["POST"])
def audio_to_audio():
    """Create Rave base audio using pre-existing audio"""
    # Get the request data
    data = request.get_json()
    input_audio_path = data["input_audio_path"]
    autoregressive_iterations = data.get("iterations", 3)
    output_path = data.get("output_path", "rave_output_new.wav")
    audio_paths = []
    for idx in range(autoregressive_iterations):
        rave_model.load_audio(input_audio_path, output_path)
        audio_paths.append(output_path)
        input_audio_path = output_path.split("wav")
        input_audio_path[0] + idx + "wav"

    return jsonify({"result": audio_paths})


if __name__ == "__main__":
    print("Starting initialization...")
    try:
        init_models()
        print("All models initialized successfully")
        print("Starting Flask server...")
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        import traceback
        traceback.print_exc()
