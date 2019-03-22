import tflearn
import datautil
import tensorflow as tf
import cv2
import numpy as np
import os

batch_size = 64
testX, testY = datautil.getBatchPrecision(batch_size)

# custom mean absolute loss function
# would prefer to use log cosh loss, but getting NaN error
def custom_loss(y_pred, y_true):
    with tf.name_scope("MeanAbsoluteError"):
        return tf.reduce_mean(tf.abs(y_pred - y_true))

tf.reset_default_graph()
sess = tf.Session()

network = tflearn.input_data([None, 50, 50, 3], name='input')
network = tflearn.conv_2d(network, 3, 3, activation='leaky_relu', name="small_CONV")
network = tflearn.conv_2d(network, 3, 5,strides=[1,1,1,1], activation='leaky_relu',name="large_CONV")
network = tflearn.conv_2d(network, 1, 5,strides=[1,1,1,1], activation='leaky_relu',name="larger_CONV")
network = tflearn.reshape(network, (-1, 50, 50), name ="final")
#no FULLY CONNECTED!
network = tflearn.regression(network, optimizer='adam', learning_rate=0.01,
                     loss='mean_square', name='target')
model = tflearn.DNN(network, tensorboard_verbose=3, max_checkpoints=3, tensorboard_dir="./logs")

print("Loading Model")
#model.load("current-pres.model")
for x in range(1000):
    X, Y = datautil.getBatchPrecision(batch_size)
    model.fit({'input': X}, {'target': Y}, n_epoch=32,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id="Precision-19-03-22")
    model.save("current-pres.model")