#  This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2023
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Homework 3


import numpy as np
import matplotlib.pyplot as plt
import utils
from skimage import data
from detectCorners import detectCorners, simple_score, harris_score

#I = data.checkerboard()
#I = utils.imread('../data/polymer-science-umass.jpg')
I = utils.imread('../data/d.png')

print(I.shape)
plt.figure(1)
cx, cy, cs, c1 = detectCorners(I, True, 1.5, 0.005)
plt.subplot(121)
#from IPython import embed; embed(); exit(-1)
if I.ndim == 2:
    plt.imshow(I, cmap='gray')
else:
    plt.imshow(I)
plt.plot(cx, cy, 'r.')
plt.title('Simple Corners')
plt.axis('off')

cx, cy, cs, c2 = detectCorners(I, False, 1.5, 0.0001)
plt.subplot(122)
if I.ndim == 2:
    plt.imshow(I, cmap='gray')
else:
    plt.imshow(I)
plt.plot(cx, cy, 'g.')
plt.title('Harris Corners')
plt.axis('off')

#v1 = simple_score(I,1.5)
#v2 = harris_score(I,1.5)
#print(v1.shape)
#plt.imshow(v1)
#plt.imshow(v2)]

plt.figure(3)
plt.imshow(c1)
plt.figure(4)
plt.imshow(c2)

plt.show()

