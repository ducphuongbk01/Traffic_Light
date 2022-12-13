import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
from utils import getFileName, getTXTDetail
from tqdm import tqdm

# Model Class
class TrafficLightNetModel():
  def __init__(self, input_shape, num_classes, size_se):
    super(TrafficLightNetModel, self).__init__()

    self.input_shape = input_shape
    self.num_classes = num_classes
    
    #Input block
    input = keras.Input(shape=self.input_shape)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Set aside residual
    previous_block_activation = x  

    for size in [32, 64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        # Add back residual
        x = layers.add([x, residual])  
        # Set aside next residual
        previous_block_activation = x  

    x = layers.SeparableConv2D(size_se, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    prev_x = x

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(size_se//2, activation="relu")(x)
    x = layers.Dense(size_se, activation="sigmoid")(x)

    x = tf.reshape(x, [-1, 1, 1, size_se])

    x = x * prev_x

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    output = layers.Dense(self.num_classes, activation="sigmoid")(x)

    self.model =  keras.Model(input, output)

    # print(self.model.summary())

  def load_model(self, model_path):
    try:
      self.model.load_weights(model_path)
      return True
    except:
      return False

  def imageProcessing(self, img, input_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.cast(img, tf.float32)
    img /= 255.0
    img = tf.image.resize(img, size=input_size)
    return img

  def predict(self, img, conf_thresh=[0.5, 0.5, 0.5], single_lb=False, verbose=0):
    labels = []
    img_pred = self.imageProcessing(img, (75, 75))

    pred = self.model.predict(np.array([img_pred]), verbose=verbose)

    if single_lb:
      max_idx = tf.math.argmax(pred[0])
      if max_idx==0:
        labels = 'r'
      elif max_idx==1:
        labels = 'y'
      elif max_idx==2:
        labels = 'g'
      else:
        labels = 'n'
    else:
      if pred[0,0] >= conf_thresh[0]:
        labels.append('r')
      if pred[0,1] >= conf_thresh[1]:
        labels.append('y')
      if pred[0,2] >= conf_thresh[2]:
        labels.append('g')
      if pred[0,0] < conf_thresh[0] and pred[0,1] < conf_thresh[1] and pred[0,2] < conf_thresh[2]:
        labels.append('n')

    conf_scores = pred[0]

    return labels, conf_scores

  def train_model(self, path_save, x_train, y_train, epochs, batch_size, x_val, y_val, verbose=1):
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=["accuracy"])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=path_save,
                                                      verbose=1,
                                                      monitor="val_accuracy",
                                                      save_best_only=True)
    
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
                 tf.keras.callbacks.TensorBoard(log_dir="logs"),
                 checkpointer]

    # print(self.model.summary())

    results = self.model.fit(
                              x_train, 
                              y_train, 
                              batch_size,
                              epochs,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_split=0.1,
                              validation_data=(x_val, y_val),
                              shuffle=True
                              )
    
    print("Finished training")
    return results

  def train_model(self, path_save, ds_train, epochs, batch_size, ds_val, verbose=1):
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=["accuracy"])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=path_save,
                                                      verbose=1,
                                                      monitor="val_accuracy",
                                                      save_best_only=True)
    
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
                 tf.keras.callbacks.TensorBoard(log_dir="logs"),
                 checkpointer]

    results = self.model.fit(ds_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=ds_val)
    print("Finished training")

    return results
    
    

  # def evaluate_model(self, test_datapath:str, conf_thresh=[0.5, 0.5, 0.5], single_lb=False):
  #   test_data_path_lst = glob(test_datapath + "/*/*.jpg")

  #   labels = []
  #   predicts = []
  #   conf_scores = []

  #   for i, path in enumerate(tqdm(test_data_path_lst)):
  #     img = cv2.imread(path)
  #     pred, score = self.predict(img, conf_thresh=conf_thresh, single_lb=single_lb)
  #     labels.append(path.split('\\')[-2])
  #     predicts.append(pred)
  #     conf_scores.append(score)

  #   return labels, predicts, conf_scores


  def evaluate_model(self, test_datapath:str, conf_thresh=[0.5, 0.5, 0.5], single_lb=False):
    test_data_path_lst = glob(test_datapath + "/image/*.jpg")
    test_label_path = test_datapath + "/gt/"

    labels = []
    predicts = []
    conf_scores = []

    for i, path in enumerate(tqdm(test_data_path_lst)):
      name = getFileName(path)
      img = cv2.imread(path)
      pred, score = self.predict(img, conf_thresh=conf_thresh, single_lb=single_lb)
      labels.append(getTXTDetail(test_label_path + name + ".txt"))
      predicts.append(pred)
      conf_scores.append(score)

    return labels, predicts, conf_scores

  def model_summary(self):
    print(self.model.summary())

  def save(self, path):
    self.model.save(path)