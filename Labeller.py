from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
import skimage.measure

start = 867
x = start
for i in range(2000-start):
    label = np.zeros((120, 280), dtype=np.uint8)
    root = Tk()
    #setting up a tkinter canvas
    w = Canvas(root, width=1120, height=480)

    #adding the image
    original = Image.open("./SmallData/"+str(x)+".jpg")
    original = original.resize((1120,480)) #resize image
    img = ImageTk.PhotoImage(original)
    w.create_image(0, 0, image=img, anchor="nw")
    w.pack()

    def writecoords(event):
        # outputting x and y coords to console
        label[event.y//4][event.x//4] = 255 # white pixel where hand is
        print("Hand at "+str(event.y//4)+" "+str(event.x//4))

    w.bind("<Button 1>",writecoords)

    root.mainloop()
    cv2.imwrite("Labels/"+str(x)+".jpg", label)
    tiny = skimage.measure.block_reduce(label, (10, 10), np.max)
    cv2.imwrite("./TinyLabels/" + str(x) + ".jpg", tiny)
    print("Labelled image "+str(x))
    x = x + 1