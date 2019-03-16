import os
import numpy as np
import cv2
import random


def getBatchRange(batch_size):
    fileList = os.listdir("./TinyLabels")
    i=random.randint(0,len(fileList)-1)
    image = cv2.imread("./SmallData/"+str(i)+".jpg")
    images = np.reshape(image, (1, 120, 280, 3))
    label = cv2.imread("./TinyLabels/"+str(i)+".jpg", 0)
    labels = np.reshape(label, (1, 12, 28))
    for x in range(batch_size-1):
        i = random.randint(0, len(fileList) - 1)
        image = cv2.imread("./SmallData/"+str(i)+".jpg")
        image = np.reshape(image, (1, 120, 280, 3))
        label = cv2.imread("./TinyLabels/" + str(i) + ".jpg", 0)
        label = np.reshape(label, (1, 12, 28))
        labels = np.vstack((labels, label))
        images = np.vstack((images, image))
    return  images, labels