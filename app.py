#!/usr/bin/env python

from flask import Flask, jsonify, request, abort
import requests
from riffusion_layers import TextLayer
from rave import Raven
import clip
from diffusers import DiffusionPipeline
import os


model, preprocess = clip.load("ViT-L/14")
diffusion_pipeline = DiffusionPipeline.from_pretrained(
    "riffusion/riffusion-model-v1"
).to("cuda")
## Riffusion
riffusion_model = None

##Rave
rave_model = Raven()


app = Flask(__name__, static_folder="./reactapp/build", static_url_path="/")


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
    app.run()
