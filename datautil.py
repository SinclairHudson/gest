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

# this allows the user to quickly check the batch to make sure everything is in order/labels are aligned.
def checkBatch(batch):
    print(batch[0].shape)
    print(batch[1].shape)
    for x in range(len(batch[0])):
        cv2.imshow(str(x), batch[0][x])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow(str(x), batch[1][x])
        cv2.waitKey(0)
        cv2.destroyAllWindows()