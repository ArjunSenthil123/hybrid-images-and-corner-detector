#  This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2023
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Homework 3

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
from scipy.ndimage import gaussian_filter
from scipy import ndimage, datasets
from utils import imread

im = imread('../data/peppers.png')
noise1 = imread('../data/peppers_g.png')
noise2 = imread('../data/peppers_sp.png')

error1 = ((im - noise1)**2).sum()
error2 = ((im - noise2)**2).sum()

print('Input, Errors: {:.2f} {:.2f}'.format(error1, error2))

plt.figure(1)

plt.subplot(131)
plt.imshow(im, cmap="gray")
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1, cmap="gray")
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(noise2, cmap="gray")
plt.title('SE {:.2f}'.format(error2))



# Denoising algorithm (Gaussian filtering)
g1 = gaussian_filter(noise1, sigma=1.25)
g2 = gaussian_filter(noise2, sigma=1.57)
error1 = ((im - g1)**2).sum()
error2 = ((im - g2)**2).sum()
print('Input, Errors: {:.2f} {:.2f}'.format(error1, error2))

plt.figure(2)

plt.subplot(131)
plt.imshow(im, cmap="gray")
plt.title('Input')

plt.subplot(132)
plt.imshow(g1, cmap="gray")
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(g2, cmap="gray")
plt.title('SE {:.2f}'.format(error2))


# Denoising algorithm (Median filtering)
m1 = ndimage.median_filter(noise1, size=5)
m2 = ndimage.median_filter(noise2, size=3)

error1 = ((im - m1)**2).sum()
error2 = ((im - m2)**2).sum()
print('Input, Errors: {:.2f} {:.2f}'.format(error1, error2))

plt.figure(3)

plt.subplot(131)
plt.imshow(im, cmap="gray")
plt.title('Input')

plt.subplot(132)
plt.imshow(m1, cmap="gray")
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(m2, cmap="gray")
plt.title('SE {:.2f}'.format(error2))


plt.show()