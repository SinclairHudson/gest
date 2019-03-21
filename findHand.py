import tflearn
import datautil
import tensorflow as tf
import cv2
import numpy as np
import os

#something's wrong. Still averaging 10% accuracy on a pretty simple task...
batch_size = 64
testX, testY = datautil.getBatchRange(batch_size)

# custom mean absolute loss function
# would prefer to use log cosh loss, but getting NaN error
def custom_loss(y_pred, y_true):
    with tf.name_scope("MeanAbsoluteError"):
        return tf.reduce_mean(tf.abs(y_pred - y_true))

tf.reset_default_graph()
sess = tf.Session()

network = tflearn.input_data([None, 120, 280, 3], name='input')
network = tflearn.conv_2d(network, 3, 5, activation='leaky_relu', name="small_CONV")
network = tflearn.conv_2d(network, 3, 5,strides=[1,2,2,1], activation='leaky_relu',name="large_CONV")
network = tflearn.conv_2d(network, 1, 5,strides=[1,5,5,1], activation='leaky_relu',name="larger_CONV")
network = tflearn.reshape(network, (-1, 12, 28), name ="final")
#no FULLY CONNECTED!
network = tflearn.regression(network, optimizer='adam', learning_rate=0.01,
                     loss='mean_square', name='target')
model = tflearn.DNN(network, tensorboard_verbose=3, max_checkpoints=3, tensorboard_dir="./logs")

print("Loading Model")
model.load("current.model")
for x in range(1000):
    X, Y = datautil.getBatchRange(batch_size)
    model.fit({'input': X}, {'target': Y}, n_epoch=32,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id="Ranger-19-03-20")
    model.save("current.model")

    image = cv2.imread("./SmallData/1000.jpg")
    image = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(image)[0]
    cv2.imwrite("1000/"+str(x)+".jpg", out)

    image = cv2.imread("./SmallData/1919.jpg")
    image = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(image)[0]
    cv2.imwrite("1919/" + str(x) + ".jpg", out)

    image = cv2.imread("./SmallData/225.jpg")
    image = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(image)[0]
    cv2.imwrite("225/" + str(x) + ".jpg", out)

    image = cv2.imread("./SmallData/451.jpg")
    image = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(image)[0]
    cv2.imwrite("451/" + str(x) + ".jpg", out)
