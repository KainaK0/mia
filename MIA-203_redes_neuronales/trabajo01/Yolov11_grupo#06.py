# %%
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="1JyiLkbtwZmSoXCqXEnc")
project = rf.workspace("grupo-6-placas").project("peru-plate-numbers")
version = project.version(3)
dataset = version.download("yolov11")
                

# %%
from ultralytics import YOLO
import os

# Load a model
model = YOLO("yolo11n.pt")

cwd = os.getcwd()
rel_path = r"Peru-Plate-Numbers-3\data.yaml"
# Train the model
train_results = model.train(
    data= os.path.join(cwd,rel_path),  # path to dataset YAML Peru-Plate-Numbers-3\data.yaml
    batch=-1,
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()
          

# %%
# # Load model
# model = YOLO(r"runs\detect\train4\weights\best.pt")
# metrics = model.val()

# %%
# Perform object detection on an image
results = model(r"Peru-Plate-Numbers-3\valid\images\20231009_202048_jpg.rf.1a67b5b5f05ba6ea31d332fdfcc9c879.jpg")
results[0].show()

# %%
# Perform object detection on an image
results = model(r"C:\Users\kainak0\Documents\gitProjects\mia\MIA-203_redes_neuronales\Peru-Plate-Numbers-3\valid\images\Foto-Placa-453-_jpg.rf.74c2435a10f0ac4b5ec258fff82f50dc.jpg")
results[0].show()

# %%
# Perform object detection on an image
results = model(r"C:\Users\kainak0\Documents\gitProjects\mia\MIA-203_redes_neuronales\Peru-Plate-Numbers-3\valid\images\Foto-Placa-505-_jpg.rf.5c1eaaeebfeefb8c38469e7adfd0a0b2.jpg")
results[0].show()


