import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import ViTModel, ViTFeatureExtractor
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from PIL import Image
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

def load_tokenizer(tokenizer_path):
  with open(tokenizer_path, 'r') as f:
      tokenizer_data = json.load(f)
  tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))
  return tokenizer

# Load the trained ViT model and feature extractor
def load_vit_model_and_processor():
  model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
  model.eval() 
  feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
  return model, feature_extractor

# Preprocess image using ViT
def preprocess_image(image_path, feature_extractor):
  image = Image.open(image_path).convert("RGB")
  inputs = feature_extractor(images=image, return_tensors="pt")
  return inputs["pixel_values"]

# Extract features using ViT
def extract_features(image_path, vit_model, feature_extractor):
  pixel_values = preprocess_image(image_path, feature_extractor)
  with torch.no_grad():
      outputs = vit_model(pixel_values)
      features = outputs.last_hidden_state.mean(dim=1).numpy()  # Global average pooling
  return features

def generate_caption(model, tokenizer, image_features, max_len):
  # Initialize the caption with the <start> token
  caption = "<start>"
  
  # Generate caption one word at a time
  for _ in range(max_len - 1):  # Generate up to max_len - 1 tokens
      # Convert the current caption to a sequence
      sequence = tokenizer.texts_to_sequences([caption])[0]
      
      # Pad the sequence to max_len - 1
      sequence = pad_sequences([sequence], maxlen=max_len - 1, padding='post')
      
      # Predict the next word
      predictions = model.predict([image_features, sequence], verbose=0)
      
      # Get the index of the word with the highest probability
      next_word_idx = np.argmax(predictions[0, -1, :])
      next_word = tokenizer.index_word.get(next_word_idx, "<unk>")
      
      # Stop if the <end> token is generated
      if next_word == "<end>":
          break
      
      # Append the predicted word to the caption
      caption += ' ' + next_word
  
  # Remove the <start> token and return the generated caption
  return caption.replace("<start>", "").strip()

# Main function to generate a caption for an image
def caption_image(image_path, model_path, tokenizer_path, max_len):
  # Load the tokenizer and model
  tokenizer = load_tokenizer(tokenizer_path)
  model = load_model(model_path)
  
  # Load the ViT model and feature extractor
  vit_model, feature_extractor = load_vit_model_and_processor()
  
  # Extract image features
  image_features = extract_features(image_path, vit_model, feature_extractor)
  
  # Generate the caption
  caption = generate_caption(model, tokenizer, image_features, max_len)
  return caption

def predict_caption(file_name):
  model_path = f"{BASE_DIR}/Vit_LSTM_6.h5"
  tokenizer_path = f"{BASE_DIR}/tokenizer_6.json"
  image_path = f"{BASE_DIR}/images/{file_name}"
  max_len = 20
  caption = caption_image(image_path, model_path, tokenizer_path, max_len)
  return caption



