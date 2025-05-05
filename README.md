# 🔍 CLIP-for-Image-to-Image-Search

A simple yet powerful image-to-image search engine built using [OpenAI's CLIP](https://openai.com/research/clip) model and [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity indexing. This tool allows you to find visually and semantically similar images from a dataset using a query image.

---

## 🚀 Features

- Uses CLIP (ViT-B/32) to extract image embeddings
- Indexes images using FAISS for fast similarity search
- Performs nearest neighbor search and filters using probability
- Visualizes results with `supervision`

---

## 📁 Repository Structure

.
├── CLIPApp.py # Main script
├── COCO-128-2.zip # Sample dataset used (contains /train images)
├── index.bin # Saved FAISS index
└── README.md # Project documentation (You are here!)


---

## 🧑‍💻 Requirements

- Python 3.8+
- PyTorch
- OpenAI CLIP
- FAISS
- Pillow
- OpenCV
- Supervision

### 🔧 Install Dependencies

```bash
pip install torch torchvision faiss-cpu opencv-python pillow supervision git+https://github.com/openai/CLIP.git

🛠️ How to Run

    Unzip COCO-128-2.zip to create a COCO-128-2/train/ folder with images.

    Run the main script:

python CLIPApp.py

This will:

    Load images and extract CLIP embeddings.

    Index them with FAISS and save to index.bin.

    Run similarity search for a given query image.

    Display the query and matched results using a probability filter.

📷 Sample Use Case

Given an image like 000000000650_jpg.rf.1b74ba165c5a3513a3211d4a80b69e1c.jpg, the model will:

    Compute its embedding.

    Find the top 10 most similar images.

    Filter results based on a thresholded probability (via exponential distance decay).

📌 Notes

    Modify the query image by changing the to_compare filename inside the script.

    You can replace the dataset with your own by updating DATASET_PATH.

🤝 Contribution

Feel free to fork this repo, use it in your own projects, or suggest improvements via Issues or Pull Requests!
📬 Contact

Created by @sumitambre
If you find this project useful, give it a ⭐️!


---

Let me know if you want to:
- Add a visual demo (like a `results.png`).
- Make it more academic/paper-style.
- Include a live Colab or HuggingFace demo link.

Would you like me to help you upload this directly as your repo README?
