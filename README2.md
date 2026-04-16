# Video Violence Detection using Deep Learning

## 1. Introduction

Video violence detection is a deep learning task that classifies video clips into two categories: **Violence** and **NonViolence**. This project addresses the need for automated video monitoring in surveillance, public safety, and content moderation.

Manual video monitoring is inefficient and error-prone, especially when large volumes of footage must be reviewed. This project proposes a deep learning pipeline that captures both spatial scene information and temporal action patterns across video frames.

## 2. Motivation

The goal is to build a robust violence detection system that can handle:
- Diverse environments and lighting conditions
- Different forms of violence, such as fights and aggressive behavior
- Dataset variability across multiple video sources

Key design principles:
- Centralize configuration for reproducible preprocessing and training
- Compare multiple architectures before selecting the best model
- Use pretrained models to improve accuracy and training stability

## 3. Dataset

The dataset is created by merging several public violence detection datasets from Kaggle:
- RLVS Dataset
- Hockey Fight Dataset
- Video Violence Detection Dataset

Original labels were normalized into a binary format:
- **Violence (1)**: fight scenes, aggressive actions
- **NonViolence (0)**: normal, non-aggressive activities

The final dataset contains around **7,000 video clips** spanning sports fights, real-world scenes, and surveillance-style footage.

## 4. Data Preparation and Pipeline

The preprocessing pipeline is designed for consistency and speed.

### 4.1 Configuration
All parameters are stored in `configs/config.py`, including:
- dataset paths
- number of frames per video
- frame resolution
- batch size

This makes it easy to reproduce experiments and swap model settings.

### 4.2 Dataset Splitting
The dataset is split into:
- 70% training
- 15% validation
- 15% testing

A custom split function preserves class balance and shuffles the data before saving it into separate directories for each class.

### 4.3 Frame Extraction
Each video is processed using OpenCV to:
- read video frames
- sample a fixed number of frames uniformly
- resize frames to a standard resolution
- normalize pixel values using ImageNet statistics

### 4.4 Error Handling and Caching
Corrupted or unreadable videos are detected and skipped automatically to avoid runtime failures.

To accelerate training, frames are cached as PyTorch tensors (`.pt` files), reducing disk I/O and improving data loading performance.

### 4.5 Batch Format
The training loader produces batches of shape:

```text
[B, NUM_FRAMES, 3, H, W]
```

This shape preserves both spatial and temporal dimensions for video model input.

## 5. Baseline Models

Three main architectures were explored before choosing the final model.

### 5.1 Transformer-Based Model (SimpleViT)

- Treats videos as sequences of frame embeddings
- Projects each frame into a latent space
- Uses a Transformer encoder to model temporal dependencies
- Aggregates the output with mean pooling

This model is simple but trained from scratch, yielding moderate accuracy around **65%**.

### 5.2 CNN-Based Model (ResNet / MobileNet)

- Uses pretrained CNNs to extract spatial features from each frame
- Pools features across time using average pooling
- Classifies the aggregated result

This approach benefits from pretrained image features and achieves better accuracy, typically **70–80%**.

### 5.3 Proposed Hybrid Model

The final recommended model combines:
- a **3D convolutional backbone** (`R3D-18`) for spatio-temporal feature extraction
- a **Transformer encoder** for learning temporal dependencies across frame embeddings

Workflow:
1. Extract features with the 3D CNN backbone
2. Remove the final classifier from the backbone
3. Pass embeddings into a Transformer encoder
4. Apply a linear classifier to predict violence or non-violence

Improvements applied:
- Use pretrained weights for the 3D CNN
- Fine-tune later layers of the backbone
- Apply weighted loss to address class imbalance

## 6. Evaluation and Results

The final hybrid model was evaluated on the test set with strong results:
- **Accuracy:** 91.43%
- **Precision:** 91.91%
- **Recall:** 90.86%
- **F1-score:** 91.38%
- **ROC-AUC:** 0.9728

Confusion matrix summary:
- True Positives: 477
- True Negatives: 483
- False Positives: 42
- False Negatives: 48

The evaluation curves show stable convergence and strong discriminative ability, making the model practical for real-world surveillance and safety systems.

## 7. Streamlit Application

A Streamlit app is included to demonstrate the model in practice.

### App features:
- Upload a video file
- Run the full preprocessing pipeline automatically
- Display a prediction label: **Violence** or **NonViolence**
- Show a confidence score for the prediction

This app provides a prototype deployment interface for testing the system interactively.

## 8. Discussion

The hybrid model outperforms the baseline Transformer-only and CNN-only approaches by combining both spatial and temporal modeling.

Key advantages:
- Better motion pattern recognition
- Stronger generalization across diverse video sources
- Stable training and faster inference using cached data

Remaining limitations:
- Higher computational cost compared to simple image-only models
- Occasional false-positive / false-negative predictions in ambiguous clips

## 9. Conclusion

This project delivers a complete deep learning pipeline for video violence detection, from data preparation to model deployment.

The final hybrid architecture achieved over **91% accuracy**, demonstrating that combining a 3D CNN backbone with Transformer temporal modeling is effective for this task.

Future work could include:
- training on larger datasets
- exploring more advanced video architectures
- optimizing the model for real-time deployment

---

## 10. How to Use This Repository

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit app

```bash
streamlit run app/app.py
```

### Step 3: Upload a video and review the prediction

The app will preprocess the uploaded video and display the predicted violence label with confidence.

### Step 4: Explore the notebooks

Use the included notebooks for data preparation, training, and evaluation:
- `notebooks/01-1-data-preparation-and-vit-model-baseline(simple ).ipynb`
- `notebooks/01-2-data-preparation-and-vit-model-baseline (ResNet).ipynb`
- `notebooks/02-training-pipeline.ipynb`
- `notebooks/02-training-pipeline-evaluation.ipynb`
