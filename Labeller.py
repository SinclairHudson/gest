from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tkinter.simpledialog

for x in range(2000):
    coordinates = []
    xcord = -1
    ycord = -1
    xcord2 = -1
    ycord2 = -1

    root = Tk()

    def continuer():
        print("C")

    #setting up a tkinter canvas
    w = Canvas(root, width=1120, height=480)

    #adding the image
    original = Image.open("./SmallData/"+str(x)+".jpg")
    original = original.resize((1120,480)) #resize image
    img = ImageTk.PhotoImage(original)
    w.create_image(0, 0, image=img, anchor="nw")
    b = Button(w, text="Submit", command=continuer)
    w.pack()

    def printcoords(event):
        # outputting x and y coords to console
        coordinates.append([event.x, event.y])
        print(event.x, event.y)

    w.bind("<Button 1>",printcoords)


    root.mainloop()
    print(coordinates)