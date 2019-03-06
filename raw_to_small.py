import numpy as np
import cv2
import os

for x in range(2373):
    image = cv2.imread("./RawData/"+str(x)+".jpg")
    small = cv2.resize(image, (280,120))
    cv2.imwrite("SmallData/"+str(x)+".jpg", small)
print("Done!")
