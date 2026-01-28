import torch
from ultralytics import YOLO

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(r'D:\seeker_algo\yolov8n.pt')  # Load the YOLO model
    
    model.train(
        data=r'C:\Users\MSI\Downloads\dataset\9k tankdatasetplus 2484 unity data\data.yaml', 
        epochs=100, 
        imgsz=640, 
        device=device
    )
