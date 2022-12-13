import tensorflow as tf
import cv2
import numpy as np
from model import TrafficLightNetModel


def load_weights_transfer(model_old, model_new, num_skip_layers):
    num_layers = len(model_old.layers)

    if num_skip_layers > num_layers:
        print("num_skip_layers must be less than number of model_old's layers")
        return False
    else:
        for i in range(0, num_layers-num_skip_layers):
            model_new.layers[i].set_weights(model_old.layers[i].get_weights())
        return True



def main():
    model_old = tf.keras.models.load_model("./new_best_model_0.h5")

    model_new = TrafficLightNetModel((75, 75, 3), 4, 512)

    load_weights_transfer(model_old, model_new.model, 11)


if __name__ == "__main__":
    main()