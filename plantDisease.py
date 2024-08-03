from PIL import Image
import numpy as np
from pathlib import Path

path_healthy = Path("PlantVillage/Pepper__bell___healthy")
path_unhealthy = Path("PlantVillage/Pepper__bell___Bacterial_spot")

targets = []
inputs = []

breakpoint()
for filename in path_healthy.iterdir():
    img = Image.open(filename)
    img = np.array(img).flatten()




