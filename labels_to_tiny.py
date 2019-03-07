import cv2
import numpy as np

for x in range(2373):
    image = cv2.imread("./Labels/" + str(x) + ".jpg")
    ycord = -1
    xcord = -1
    ycord2 = -1
    xcord2 = -1
    z = 1
    for i in range(120):
        for j in range(280):
            if(image[i][j]!=0):
                if z == 1:
                    ycord = i
                    xcord = j
                else:
                    ycord2 = i
                    ycord2 = j
                break
        else:
            continue
        break
    tiny = np.zeros((12,28))
    ycord=