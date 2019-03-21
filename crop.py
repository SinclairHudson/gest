import numpy as np
import cv2

for x in range(1001):
    image = cv2.imread("./TinyLabels/" + str(x) + ".jpg", cv2.IMREAD_GRAYSCALE)
    colour = cv2.imread("./SmallData/"+str(x)+".jpg")
    padded = cv2.copyMakeBorder(colour, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
    count = 0
    for y in range(12):
        for z in range(28):
            if(image[y][z] > 200):
                hand = padded[(10*(y-2))+30:(10*(y+3))+30, (10*(z-2))+30:(10*(z+3))+30] # crop
                cv2.imwrite("./Hands/"+str(x)+"-"+str(count)+".jpg", hand)
                count= count + 1


