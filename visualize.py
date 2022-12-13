# Visualize Data
import matplotlib.pyplot as plt
import random
from glob import glob
import cv2
import pandas as pd


df = pd.read_csv("/home/phuongdoan/Phuong/Traffic_Light/data_total/train/train.csv")
sub_df = df.loc[df['Filenames'] == "111_image_012800.jpg"]
print(int(sub_df['r']))

