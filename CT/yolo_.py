import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolo11n-seg.pt").to(device)

results = model.train(
    batch=8,
    device=device,
    data="data.yaml",
    epochs=40,
    imgsz=256,
    freeze=0
)
