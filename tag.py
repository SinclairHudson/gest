import numpy as np
import cv2
import os
from PIL import Image
z = 253
for i in range(2374-z):
    print("======================================================")
    label = np.zeros((120,280), dtype=np.uint8)
    numHands = int(input("How many Hands are in the image " +str(z)+ " ? "))
    if(numHands == 1):
        x = int(input("What is the X coordinate of the center of the hand? "))
        y = int(input("What is the Y coordinate of the center of the hand? "))
        label[y][x]=255 
    elif(numHands == 2):
        x = int(input("What is the X coordinate of the center of the left hand? "))
        y = int(input("What is the Y coordinate of the center of the left hand? "))
        x2 = int(input("What is the X coordinate of the center of the right hand? "))
        y2 = int(input("What is the Y coordinate of the center of the right hand? "))
        label[y][x]=255
        label[y2][x2]=255
    cv2.imwrite("Labels/"+str(z)+".jpg", label)
    cv2.destroyAllWindows()
    z = z + 1
        
