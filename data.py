import albumentations as A
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

data_transform = A.Compose([
                          A.VerticalFlip(p=0.5),
                          A.HorizontalFlip(p=0.5),
                          A.RandomSnow(p=0.5),
                          A.RandomRain(p=0.5),
                          A.augmentations.geometric.rotate.Rotate(limit=[-30,30],p=0.5),
                          A.augmentations.transforms.MotionBlur(blur_limit=7, p=0.5),
                          A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
                          ])


def generateData(data_lst:list, lb_lst:list, ratio=2):
  data_lst_tmp = data_lst.copy()
  lb_lst_tmp = lb_lst.copy()
  for i, img in enumerate(data_lst):
    for j in range(ratio-1):
      img2 = data_transform(image=img)["image"]
      data_lst_tmp.append(img2)
      lb_lst_tmp.append(lb_lst[i])
  return data_lst_tmp, lb_lst_tmp

