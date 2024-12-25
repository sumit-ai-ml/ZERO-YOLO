import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from ultralytics import YOLO
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolo11x-seg.pt").to(device)


results = model.train(
    batch=18,
    device=device,
    data="data.yaml",
    epochs=500,
    imgsz=256,
    freeze=0         # Replace with your desired run name
)
