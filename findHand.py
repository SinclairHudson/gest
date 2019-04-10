import tflearn
import datautil
import tensorflow as tf
import cv2
import numpy as np
import datetime
now = datetime.date.today()

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
network = tflearn.conv_2d(network, 3, 8, activation='leaky_relu',name="large_CONV")
network = tflearn.max_pool_2d(network, 2, name="max_POOL")
network = tflearn.conv_2d(network, 1, 5, activation='leaky_relu',name="final_CONV")
network = tflearn.avg_pool_2d(network, 5, name="avg_POOL") # want avg here, to reduce outliers.
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
              snapshot_step=100, show_metric=True, run_id="Ranger-"+str(now))
    model.save("current.model")

    #taking samples every 32 steps, and saving them in an image.
    image = cv2.imread("./SmallData/1000.jpg")
    input = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(input)[0]
    out = cv2.cvtColor(out,cv2.COLOR_GRAY2RGB)
    out = cv2.resize(out, (280, 120), interpolation=cv2.INTER_AREA)
    stack1 = np.concatenate((image, out), axis=0)

    image = cv2.imread("./SmallData/76.jpg")
    input = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(input)[0]
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    out = cv2.resize(out, (280, 120), interpolation=cv2.INTER_AREA)
    stack2 = np.concatenate((image, out), axis=0)

    image = cv2.imread("./SmallData/1427.jpg")
    input = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(input)[0]
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    out = cv2.resize(out, (280, 120), interpolation=cv2.INTER_AREA)
    stack3 = np.concatenate((image, out), axis=0)

    image = cv2.imread("./SmallData/1653.jpg")
    input = np.reshape(image, (1, 120, 280, 3))
    out = model.predict(input)[0]
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    out = cv2.resize(out, (280, 120), interpolation=cv2.INTER_AREA)
    stack4 = np.concatenate((image, out), axis=0)

    bigstack1 = np.concatenate((stack2, stack3), axis=0)
    bigstack2 = np.concatenate((stack1, stack4), axis=0)
    final = np.concatenate((bigstack1, bigstack2), axis = 1)
    cv2.imwrite("./slideshow/"+str(x+551)+".jpg", final)

