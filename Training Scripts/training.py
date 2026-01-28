from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
def main():
    model = YOLO('yolov8s.pt')
    results = model.train(data='data.yaml', epochs=300, imgsz=640, batch=32, device= 0)

if __name__ == '__main__':
    main()