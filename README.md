# gest
Gest is a system that uses machine vision to track hand gestures and location over a desk. This allows the user to interact with their computer in an incredibly intuitive and new fashion. (Work in progress)
# The execution:
Gest will first take an image from the webcam stream and scale it down for faster processing. 
Then, a CNN processes the image in an attemp to locate the general region a hand is in.
After that, the targetted region is cropped from the image, and another CNN tries to find the very center of the hand.
The same region is analysed again to classify the current gesture the hand is performing.
Finally, all the data is added back up to create a position and gesture of the hand, which can then be mapped to the computer.

#Applications