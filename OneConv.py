#trying out one convolutional neural network, since with no fully connected layers it should be possible to parse a larger image.
import tflearn
import datautil
import tensorflow as tf
import cv2
import numpy as np
import datetime
import os
now = datetime.date.today()

LR = 0.001 # learning rate
batch_size = 64
testX, testY = datautil.getBatchOneConv(batch_size)

# custom mean absolute loss function
# would prefer to use log cosh loss, but getting NaN error
def custom_loss(y_pred, y_true):
    with tf.name_scope("MeanAbsoluteError"):
        return tf.reduce_mean(tf.abs(y_pred - y_true))

tf.reset_default_graph()
sess = tf.Session()

network = tflearn.input_data([None, 120, 280, 3], name='input')
network = tflearn.conv_2d(network, 6, 4, activation='leaky_relu', name="small_CONV")
network = tflearn.conv_2d(network, 6, 10, activation='leaky_relu', name="small_CONV2")
network = tflearn.conv_2d(network, 3, 25, activation='leaky_relu',name="large_CONV")
network = tflearn.conv_2d(network, 1, 5, activation='leaky_relu',name="final_CONV")
network = tflearn.reshape(network, (-1, 120, 280), name ="final")

network = tflearn.regression(network, optimizer='adam', learning_rate=LR,
                     loss='mean_square', name='target')
model = tflearn.DNN(network, tensorboard_verbose=3, max_checkpoints=3, tensorboard_dir="./logs")

print("Loading Model")
model.load("oneconv.model")
for x in range(1000):
    X, Y = datautil.getBatchOneConv(batch_size)
    model.fit({'input': X}, {'target': Y}, n_epoch=32,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id="OneConv-"+"2019-04-09")
    model.save("oneconv.model")