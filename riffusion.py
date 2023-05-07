import locale
locale.getpreferredencoding = lambda: "UTF-8"

from torch.jit import Error
import clip 
import math
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import torch
from riffusion_inference.riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion_inference.riffusion.spectrogram_params import SpectrogramParams
from uuid import uuid4
import PIL.Image as Image


def check_valid_coordinate(coordiante):
  """simple function to check if coordinate is a tuple"""
  valid = False  
  if np.shape(coordiante)== (2,):
    valid = True
  if valid == False:
    raise ValueError("Invalid Coordinate.")
  return valid

def normalize(vector):
    """Normalize distances to add up to 1"""
    total = sum(vector)
    return np.array([x / total for x in vector])

class TextLayer:

  def __init__(self, model, preprocess, diffusion_pipeline, text_labels):
    self.model = model.to('cuda')
    self.preprocess = preprocess
    self.text_labels = text_labels
    self.text_embeddings = []
    self.embeddings_coordinates_pca = []
    self.pipe = diffusion_pipeline.to("cuda")
    self.spectral_images = []

  def text_to_embedding(self, text):
    """Take text embeded it using CLIP's text_encoder then return the text embedding and the text"""
    tokenized = clip.tokenize(text)
    text_embedding = self.model.encode_text(tokenized.to('cuda')).float()
    text_embedding = text_embedding.cpu()
    text_embedding = text_embedding.detach().numpy()
    return text,text_embedding

  def user_embedding_distance(self, mouse_location, text_coordinate):
    """The distance between the mouse cursor and a specific text embedding coordinate"""
    
    check_valid_coordinate(mouse_location)
    check_valid_coordinate(text_coordinate)
    return math.dist(mouse_location, text_coordinate)

  def all_embedding_distances(self, mouse_location):
    """Get normalized 1 Dimensional vector of distances between mouse and text_labels"""
    embedding_distances = [self.user_embedding_distance(mouse_location, text_coordinate) for text_coordinate in self.embeddings_coordinates_pca]
    
    normalized_embedding_distances = normalize(embedding_distances)
    index = np.where(normalized_embedding_distances == 1.0)[0]
    print(f"intitial embedding distances: {normalized_embedding_distances}")
    # Zero out all other elements if the element with value 0 exists
    if len(index) > 0:
      normalized_embedding_distances = np.where(normalized_embedding_distances == normalized_embedding_distances[index[0]], normalized_embedding_distances, 0)
      normalized_embedding_distances[index[0]] = 1

    return normalized_embedding_distances

  def embed_text(self):
    for text_label in self.text_labels:
      text, text_embedding = self.text_to_embedding(text_label)
      self.text_embeddings.append(text_embedding)
    print("Completed Embedding Text")


  def decomposition(self):
    """Dimensionality Reduction using PCA"""
    pca = PCA(n_components=2)
    scaler =  MinMaxScaler(feature_range=(0, 100))

    # Fit the PCA model to the data and transform the data into the principal component space
    embeddings = np.concatenate(self.text_embeddings)
    pca.fit(embeddings)

    # Transform the embeddings into the principal component space

    embeddings_coordinates_pca = pca.fit_transform(embeddings)
    embedding_coordinates_pca_normalized = scaler.fit_transform(embeddings_coordinates_pca)

    print(f"Embeddings Shape: {embedding_coordinates_pca_normalized.shape}")
    self.embeddings_coordinates_pca = embedding_coordinates_pca_normalized

  def create_inital_audio(self):
    '''Create audio spectralgrams for each of the text labels'''
    self.embed_text()
    self.decomposition()
    for text_label in self.text_labels:
      spectral_image = self.pipe(
          text_label,
          #negative_prompt=negative_prompt,# this will improve the model
          width=768,
      ).images[0] # here we are just using the first image we should use clip to rank these
      self.spectral_images.append(np.array(spectral_image))
    print("Finished Creating Intial Images")

  def create_new_encoding(self, cursor_cordinate):
    '''Create new Audio spectral image using distance vector'''
    distance_vector = self.all_embedding_distances(cursor_cordinate)
    print(distance_vector)
    resultant_embedding = np.array(text_layer.spectral_images).T * distance_vector
    resultant_embedding = np.expand_dims(np.sum(resultant_embedding.T, axis=0),axis=0)
    new_embedding = np.squeeze(resultant_embedding)
    return new_embedding

  def save_new_embedding(self, cursor_cordinate, save_path=None):
    new_encoding = self.create_new_encoding(cursor_cordinate)
    pil_image = Image.fromarray(new_encoding.astype(np.uint8))
    wav = self.audio_from_image(pil_image)
    if not save_path:
      uuid = uuid4()
      save_path = 'riffusion_output_' + str(uuid)
    save_path += '.wav'
    wav.export(save_path, format='wav')
    return save_path

  def audio_from_image(self,new_encoding, save_path=None):
    params = SpectrogramParams()
    converter = SpectrogramImageConverter(params)
    wav = converter.audio_from_spectrogram_image(image=new_encoding)
    return wav


  