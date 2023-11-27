import onnx
import torch
from ultralytics import YOLO
import os 
import onnxruntime as ort
from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
from pathlib import Path

parent_dir = "../runs/"
try:  
    os.mkdir(parent_dir)  
except OSError as error:  
    print(error)   

# Directory 
directory = "exported/"
# Parent Directory path 

# Path 
path = os.path.join(parent_dir, directory) 

# Create the directory 
# 'GeeksForGeeks' in 
# '/home / User / Documents' 
try:  
    os.mkdir(path)  
except OSError as error:  
    print(error)   


model = YOLO('./best.pt')
model.export(format="onnx")

onnx_opset = 16
from packaging import version
if version.parse(ort.__version__) >= version.parse("1.14.0"):
    onnx_opset = 17

output_format : str = "jpg"
# add the processing to the model and output a PNG format image. JPG is also valid.
add_ppp.yolo_detection(Path('./best.onnx'), Path('../runs/exported/best_preproc.onnx'), output_format, onnx_opset, num_classes=11, input_shape=[640, 640])