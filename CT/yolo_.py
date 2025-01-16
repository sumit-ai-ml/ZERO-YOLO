
from pathlib import Path
from ultralytics import YOLO
import torch

#import wandb

def yolo_train():
    #script_dir = Path(__file__).parent.resolve()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_dir = Path().resolve() 
    
    model_path = script_dir / 'runs/segment/train17/weights/best.pt'
    model = YOLO(model_path).to(device)

    script_dir = Path().resolve()
    yaml_path =script_dir / 'CT/data.yaml'
    results = model.train(
        batch=10,
        device=device,
        data=yaml_path,
        epochs=500,
        imgsz=256,
        freeze=0         # Replace with your desired run name
    )
