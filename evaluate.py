from glob import glob
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from model import TrafficLightNetModel
from utils import plot_confusion_matrix
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data import data_transform
import numpy as np


def main():
    tfModel = TrafficLightNetModel((75, 75, 3), 4, 256)

    tfModel.load_model("./tflight_mini_best_model_3.h5")

    test_datapath = "./traffic_light_4000"
    labels, predicts, conf_scores = tfModel.evaluate_model(test_datapath, conf_thresh=[0.8, 0.8, 0.8], single_lb=True)

    cnf_matrix = confusion_matrix(labels, predicts)
    plot_confusion_matrix(cnf_matrix, classes=['r', 'y', 'g', 'n'],
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues)

    acc = accuracy_score(labels, predicts)
    f1 = f1_score(labels, predicts, average='micro')
    pre = precision_score(labels, predicts, average='micro')
    re = recall_score(labels, predicts, average='micro')

    print("Finish evaluate...")
    print("Accuracy: ", acc)
    print("F1 score: ", f1)
    print("Precision: ", pre)
    print("Recall: ", re)


if __name__ == "__main__":
    main()

    