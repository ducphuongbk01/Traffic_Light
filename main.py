from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import TrafficLightNetModel
from utils import plot_confusion_matrix
import cv2

def main():
    tfModel = TrafficLightNetModel((75, 75, 3), 4)

    tfModel.load_model("./best_model5.h5")

    test_datapath = "./data/traffic_light_4000"
    labels, predicts, conf_scores = tfModel.evaluate_model(test_datapath, conf_thresh=[0.5, 0.5, 0.5], single_lb=True)

    cnf_matrix = confusion_matrix(labels, predicts)
    plot_confusion_matrix(cnf_matrix, classes=['r', 'y', 'g', 'n'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)



if __name__ == "__main__":
    main()