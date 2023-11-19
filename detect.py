import os
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
images_folder = 'datasets/test/images'

for filename in os.listdir(images_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(images_folder, filename)

        result = model.predict(image_path, save=True)
