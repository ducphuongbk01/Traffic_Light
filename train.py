from sklearn.metrics import confusion_matrix
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

    train_path = "./data/train"
    val_path = "./data/validation"

    ds_train = image_dataset_from_directory(train_path,
                                            labels='inferred',
                                            label_mode='int',
                                            class_names=['r', 'y', 'g', 'n'],
                                            color_mode='rgb',
                                            batch_size=32,
                                            image_size=(75, 75),
                                            shuffle=True,
                                            seed=123,
                                            validation_split=0.1,
                                            subset="training"                                       
                                            )

    ds_val = image_dataset_from_directory(  val_path,
                                            labels='inferred',
                                            label_mode='int',
                                            class_names=['r', 'y', 'g', 'n'],
                                            color_mode='rgb',
                                            batch_size=32,
                                            image_size=(75, 75),
                                            shuffle=False,
                                            seed=123,
                                            validation_split=0.1,
                                            subset="validation"                                       
                                            )

    def aument_data(x, y):
        img = np.array(x)
        img = data_transform(image=img)["image"]
        img = tf.convert_to_tensor(img)
        return img, y

    ds_train = ds_train.map(aument_data)

    # tfModel = TrafficLightNetModel((75, 75, 3), 4)

    # # tfModel.load_model("/content/trafficlight_classifier_v2/trafficlightfull.h5")

    # path_save = "./models/best_model_pretrain0.h5"

    # tfModel.train_model(path_save=path_save, ds_train=ds_train, epochs=100, batch_size=32, ds_val=ds_val, verbose=1)

    for images, labels in ds_train.take(1):
        print(type(images[0]))
        break

if __name__ == "__main__":
    main()