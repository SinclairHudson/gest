import tflearn
import datautil
import datetime
now = datetime.date.today()

batch_size = 64 # defining batch size and getting a batch for testing
testX, testY = datautil.getBatchClassification(batch_size)


network = tflearn.input_data([None, 50, 50, 3], name='input')
network = tflearn.conv_2d(network, 4, 5, activation='leaky_relu', name="first_CONV")
network = tflearn.conv_2d(network, 4, 5, activation='leaky_relu',name="second_CONV")
network = tflearn.max_pool_2d(network, 2, name="first_POOL") # pooling down to 25x25
network = tflearn.conv_2d(network, 3, 5, activation='leaky_relu',name="third_CONV")
network = tflearn.fully_connected(network,2, activation="leaky_relu", name="Fully_Connected")
network = tflearn.regression(network, optimizer='adam', learning_rate=0.005,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=3, max_checkpoints=3, tensorboard_dir="./logs")

print("Loading Model")
model.load("current-class.model")
for x in range(1000):
    X, Y = datautil.getBatchClassification(batch_size)
    model.fit({'input': X}, {'target': Y}, n_epoch=32,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id="Class-"+"2019-04-05")
    model.save("current-class.model")
