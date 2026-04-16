# src/load_model.py
import torch
import os
from models.model import HybridVideoClassifier
from configs.config import BASE_DIR, NUM_FRAMES

def load_trained_model(model_path="hybrid_model3.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridVideoClassifier(num_frames=NUM_FRAMES, num_classes=1, freeze_backbone=True).to(device)

    full_path = os.path.join(BASE_DIR, "models", model_path)
    model.load_state_dict(torch.load(full_path, map_location=device))

    model.to(device)
    model.eval()

    return model, device