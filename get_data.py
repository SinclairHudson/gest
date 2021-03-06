import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)

i = 2373
#main loop
while(cap.isOpened()):  # check !
    # capture frame-by-frame
    ret, frame = cap.read()
    i= i + 1
    if ret: # check ! (some webcam's need a "warmup")
        # our operation on frame come her
        print(frame.shape)
        time.sleep(0.5)
        crop_img = frame[120:360, 80:640]
        # Display the resulting frame
        cv2.imshow('frame', crop_img)
        cv2.imwrite("RawData/"+str(i)+".jpg", crop_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done release the capture
cap.release()
cv2.destroyAllWindows()
