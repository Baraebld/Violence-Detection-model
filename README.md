---
title: Violence Detection App
emoji: "🎬"
colorFrom: "blue"
colorTo: "green"
sdk: "streamlit"
sdk_version: "1.26.0"   # adjust to your Streamlit version
python_version: "3.10"
app_file: "app/app.py"
pinned: false
---

# Violence Detection in Videos 

A deep learning project for **video violence detection** using a hybrid model combining **3D CNN (ResNet-18)** for spatial-temporal feature extraction and a **Transformer** for temporal modeling. Includes a **Streamlit app** for uploading videos and classifying them as **Violent** or **Non-Violent** in real-time.

---

## Features
- **Hybrid Video Classifier**: 3D CNN backbone + Transformer for temporal modeling.
- **Pretrained Model**: Ready-to-use `hybrid_model3.pth` for inference.
- **Robust Video Processing**: Handles variable-length videos, corrupted frames, grayscale/multi-channel input.
- **Streamlit App**: Simple interface for video classification.
- **Modular Structure**: Clear separation between model, utils, configs, and app.

---

## 📁 Project Structure

Violence-Detection-video/
│
├── app/
│   ├── app.py                 # Streamlit web application
│   └── __init__.py
│
├── data/
│   └── test_videos/           # Sample videos for testing
│
├── models/
│   ├── hybrid_model3.pth      # Final trained hybrid model
│   └── model.py               # Hybrid model definition
│
├── src/
│   ├── video_utils.py         # Video preprocessing utilities
│   ├── load_model.py          # Model loading helper
│   └── __init__.py
│
├── configs/
│   ├── config.py              # Dataset, preprocessing, training configs
│   └── __init__.py
│
├── notebooks/
│   ├── 01-1-data-preparation-and-vit-model-baseline(simple).ipynb
│   ├── 01-2-data-preparation-and-vit-model-baseline(ResNet ).ipynb
│   ├── 02-training-pipeline-evaluation.ipynb
│   └── 02-training-pipeline.ipynb
│
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── .gitignore
└── LICENSE

---

## Dataset & Preprocessing
- Merged from Kaggle: violent vs non-violent videos.
- Videos processed on-the-fly in the app: extract `NUM_FRAMES`, resize, normalize.
- **Cached `.pt` datasets are only needed for training**; app does not require caching.

---

## Model Architecture
- **Backbone**: `r3d_18` (3D ResNet-18), removes classifier, frozen except optional last block.
- **Temporal Transformer**: 3 layers, `hidden_dim=512`, `nhead=8`.
- **Classification Head**: Linear layer outputs single logit with `BCEWithLogitsLoss`.
- **Forward pass**: `[B, F, C, H, W] -> Backbone -> Transformer -> Classifier -> logit`

> Note: Current model repeats backbone features across frames; works fine for deployment.

---

## Installation and Usage 
```bash
## 🔥 Usage

You can use the Violence Detection app in **two ways**:

---

### **Method 1 — Directly via Hugging Face Spaces**

- The model is deployed online using Hugging Face Spaces, allowing users to test it directly without any local setup.

[🔗 Open the App here ] https://huggingface.co/spaces/rihammsd10/violence-detection-app

- Simply upload a video and the model will return the **Violence Probability** and **Non-Violence Probability**.

---

### **Method 2 — Run Locally**

1. Clone the GitHub repository:

git clone https://github.com/riham-esi/Violence-Detection-video.git
cd Violence-Detection

2. Install required packages:

pip install -r requirements.txt

3. Download the pretrained model (hybrid_model3.pth) from:
Hugging Face: 🔗 https://huggingface.co/rihammsd10/hybrid_model/blob/main/hybrid_model3.pth

Kaggle model link: 🔗 https://www.kaggle.com/models/messaoudiriham/best-hybrid-model
4. Place the hybrid_model3.pth file in the models/ folder:
Violence-Detection/
│
├── models/
│   └── hybrid_model3.pth

5. Run the Streamlit app:
streamlit run app/app.py

Upload a video locally and view the predictions.
```
## Training (Optional)
```bash
Use the notebooks in notebooks/:
01-1-data-preparation-and-vit-model-baseline(simple).ipynb → Prepare and cache videos and train simple model
01-2-data-preparation-and-vit-model-baseline(ResNet).ipynb → Prepare and cache videos and train ResNet model.
02-training-pipeline.ipynb → Train or fine-tune the model with pretrained hybrid model.
02-training-pipeline-evaluation.ipynb →Add evaluation to the final model trained 
```
---

## Evaluation Metrics
```bash
Accuracy
Precision
Recall
F1-score
Confusion_matrix ( TP, TN, FP, FN)
```
---
## Requirements
```bash
Python 3.8+
PyTorch
Torchvision
OpenCV
Streamlit
Numpy
```