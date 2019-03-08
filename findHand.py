import tflearn
import datautil
import tensorflow as tf

testX, testY = datautil.getBatchRange(64)
print(testX.shape)
print(testY.shape)

network = tflearn.input_data([None, 280, 120, 3], name='input')
network = tflearn.conv_2d(network, 3, 5, activation='leaky_relu')
network = tflearn.conv_2d(network, 3, 20, activation='leaky_relu')
network = tflearn.fully_connected(network, 336, activation='leaky_relu')
network = tflearn.reshape(network, (64, 12, 28), name="output")
network = tflearn.regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=3)
for x in range(100):
    X, Y = datautil.getBatchRange(64)
    model.fit({'input': X}, {'target': Y}, n_epoch=32,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id='convnet')
