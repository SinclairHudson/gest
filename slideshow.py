import numpy as np
import cv2
import os
from PIL import Image
i = 253
for x in range(2374-i):
    image = cv2.imread("./SmallData/"+str(i)+".jpg")
    cv2.imshow(str(i),image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i = i + 1
