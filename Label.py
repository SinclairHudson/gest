from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
import skimage.measure
import os

fileList = os.listdir("./Hands")

for i in range(len(fileList)):
    root = Tk()
    #setting up a tkinter canvas
    w = Canvas(root, width=480, height=520)

    #adding the image
    original = Image.open("./Hands/"+fileList[i])
    original = original.resize((480,480)) #resize image
    img = ImageTk.PhotoImage(original)
    w.create_image(0, 0, image=img, anchor="nw")
    w.pack()


    def callback():
        os.rename("./Hands/"+fileList[i],"./Hands/"+fileList[i][0:-4]+"-OPEN.jpg")
        print("./Hands/" + fileList[i][0:-4] + "-OPEN.jpg")
        root.destroy()
    def closed():
        os.rename("./Hands/"+fileList[i],"./Hands/"+fileList[i][0:-4]+"-FIST.jpg")
        print("./Hands/"+fileList[i][0:-4]+"-FIST.jpg")
        root.destroy()



    open = Button(root, text="neutral", command=callback)
    fist = Button(root, text="fist", command=closed)
    fist.pack()
    open.pack()

    root.mainloop()