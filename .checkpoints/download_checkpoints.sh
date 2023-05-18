#!/bin/bash

# Google Drive file ID
FILE_ID="1oruqzflcmPRTdLx71jt7BNgQ_SKx_7Dc"
# Desired file name and extension
FILE_NAME="rave_pretrained.ckpt"

FILE_ID2="1yPYuF3vk1Ge-0NwX0tP3lOJyfjNKdwW7"
FILE_NAME2="latest_rave_pretrained.ckpt"

# Create the .checkpoints directory if it doesn't exist
mkdir -p .checkpoints

# Download the file from Google Drive and save it in the .checkpoints folder
wget --no-check-certificate "https://drive.google.com/uc?id=${FILE_ID}&export=download" -O ".checkpoints/${FILE_NAME}"
wget --no-check-certificate "https://drive.google.com/uc?id=${FILE_ID2}&export=download" -O ".checkpoints/${FILE_NAME2}"

current_dir=$(pwd)

export PYTHONPATH="${PYTHONPATH}:$current_dir"
riffusion_inference_path = current_dir + 'riffusion_inference'
export PYTHONPATH="${PYTHONPATH}:$riffusion_inference_path"

