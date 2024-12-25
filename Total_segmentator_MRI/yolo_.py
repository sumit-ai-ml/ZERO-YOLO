
from pathlib import Path
from ultralytics import YOLO
import torch

#import wandb

def yolo_train():
    script_dir = Path(__file__).parent.resolve()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = script_dir / 'yolo11x-seg.pt'
    model = YOLO(model_path).to(device)


    results = model.train(
        batch=18,
        device=device,
        data="data.yaml",
        epochs=500,
        imgsz=256,
        freeze=0         # Replace with your desired run name
    )
