from ultralytics import YOLO
import torch
import shutil

if __name__ == '__main__':
    runs_folder = 'runs'
    shutil.rmtree(runs_folder, ignore_errors=True)

    device = '0' if torch.cuda.is_available() else 'cpu'
    if device == '0':
        torch.cuda.set_device(0)

    model = YOLO('yolov8n.pt', task='detect')

    results = model.train(data='data.yaml', epochs=10, mode='train', imgsz=640, name='train')

    print(results)
