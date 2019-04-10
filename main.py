import numpy as np
import cv2
import time
import tflearn
cap = cv2.VideoCapture(0)


#initializing the first locator network
network = tflearn.input_data([None, 120, 280, 3], name='input')
network = tflearn.conv_2d(network, 6, 4, activation='leaky_relu', name="small_CONV")
network = tflearn.conv_2d(network, 6, 10, activation='leaky_relu', name="small_CONV2")
network = tflearn.conv_2d(network, 3, 25, activation='leaky_relu',name="large_CONV")
network = tflearn.conv_2d(network, 1, 5, activation='leaky_relu',name="final_CONV")
network = tflearn.reshape(network, (-1, 120, 280), name ="final")

network = tflearn.regression(network, optimizer='adam', learning_rate=0.001,
                     loss='mean_square', name='target')
model = tflearn.DNN(network, tensorboard_verbose=3, max_checkpoints=3, tensorboard_dir="./logs")

print("Loading Model")
model.load("oneconv.model")


#main loop
while(cap.isOpened()):  # check !
    # capture frame-by-frame
    ret, frame = cap.read()

    if ret: # check ! (some webcam's need a "warmup")
        crop_img = frame[120:360, 80:640]
        # Display the resulting frame
        cv2.imshow('frame', crop_img)
        crop_img = cv2.resize(crop_img,(280,120))
        crop_img = np.reshape(crop_img, (1,120,280,3))
        out = model.predict(crop_img)[0]
        cv2.imshow('out', out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done release the capture
cap.release()
cv2.destroyAllWindows()
