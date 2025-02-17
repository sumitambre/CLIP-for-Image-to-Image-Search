import base64
import os
from io import BytesIO
import cv2
import faiss
import numpy as np
import torch
import clip
from PIL import Image
import json
import supervision as sv

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to get image embedding using local CLIP model
def get_image_embedding(image: Image) -> np.ndarray:
    """Get image embedding using local OpenAI CLIP model"""
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy().flatten()

# Initialize FAISS index
index = faiss.IndexFlatL2(512)
file_names = []

DATASET_PATH = "COCO-128-2"
TRAIN_IMAGES = os.path.join(DATASET_PATH, "train")

for frame_name in os.listdir(TRAIN_IMAGES):
    try:
        frame = Image.open(os.path.join(TRAIN_IMAGES, frame_name))
    except IOError:
        print("Error loading:", frame_name)
        continue

    embedding = get_image_embedding(frame)
    index.add(np.array([embedding]).astype(np.float32))
    file_names.append(frame_name)

faiss.write_index(index, "index.bin")
print("FAISS Index saved successfully!")

# Run an image search query
to_compare = "000000000650_jpg.rf.1b74ba165c5a3513a3211d4a80b69e1c.jpg"
to_compare_path = os.path.join(TRAIN_IMAGES, to_compare)
query_embedding = get_image_embedding(Image.open(to_compare_path))
D, I = index.search(np.array([query_embedding]).astype(np.float32), 10)  # Search for top 10 results

# Convert distances to probabilities
probabilities = np.exp(-D[0])  # Exponential function for probability conversion

# Filter results based on probability threshold
threshold = 0.  
filtered_images = [cv2.imread(os.path.join(TRAIN_IMAGES, file_names[i])) for i, prob in zip(I[0], probabilities) if prob > threshold]

# Display query image
query_image = cv2.imread(to_compare_path)
sv.plot_image(query_image, (5, 5))

# Display filtered matched images
if filtered_images:
    sv.plot_images_grid(filtered_images, (4,4))
else:
    print("No matches found with probability > 0.5")
