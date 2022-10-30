# Visualize Data
from importlib.resources import path
import matplotlib.pyplot as plt
import random
import cv2
from glob import glob
import random

data_y = glob("./data_new_2/validation/y/*.jpg")
data_rand = random.sample(data_y, k=25)

for i, path in enumerate(data_rand):
    ax = plt.subplot(5, 5, i + 1)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow((img).astype("uint8")) 
    plt.axis("off")

plt.show()