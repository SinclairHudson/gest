import tflearn
import datautil
import tensorflow as tf
import cv2
import numpy as np
import os

#something's wrong. Still averaging 10% accuracy on a pretty simple task...
batch_size = 64
testX, testY = datautil.getBatchRange(batch_size)

tf.reset_default_graph()
sess = tf.Session()

network = tflearn.input_data([None, 120, 280, 3], name='input')
network = tflearn.conv_2d(network, 3, 5, activation='leaky_relu', name="small_CONV")
network = tflearn.conv_2d(network, 3, 10,strides=[1,2,2,1], activation='leaky_relu',name="large_CONV")
network = tflearn.conv_2d(network, 1, 20,strides=[1,5,5,1], activation='leaky_relu',name="larger_CONV")
network = tflearn.reshape(network, (-1, 12, 28), name ="final")
#no FULLY CONNECTED!
network = tflearn.regression(network, optimizer='adam', learning_rate=0.1, # <--impatient
                     loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=3, max_checkpoints=3)

print("Loading Model")
#model.load("current.model")
#os.system("tensorboard --logdir=tmp/tflearn_logs/") #start tensorboard
for x in range(500):
    X, Y = datautil.getBatchRange(batch_size)
    model.fit(X, Y, n_epoch=32,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True)
    model.save("current.model")

    image = cv2.imread("./SmallData/1000.jpg")
    image = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(image)[0]
    cv2.imwrite("out.jpg", out)
