#This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2023
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Homework 3

from scipy import ndimage, datasets
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from nms import nms

from utils import gaussian
import matplotlib.pyplot as plt

def detectCorners(I, is_simple, w, th):
#Convert to float
    I = I.astype(float)

    #Convert color to grayscale
    if I.ndim > 2:
        I = rgb2gray(I)

    # Step 1: compute corner score
    if is_simple:
        corner_score = simple_score(I, w)
    else:
        corner_score = harris_score(I, w)

    
    c = corner_score

    # Step 2: Threshold corner score and find peaks
    corner_score[corner_score < th] = 0

    cx, cy, cs = nms(corner_score)
    return cx, cy, cs, c


#--------------------------------------------------------------------------
#                                    Simple score function (Implement this)
#--------------------------------------------------------------------------
def f(u,v):
    if(u,v == (1,1)): return np.array([[0, 0, 0],[0, -1, 0],[0, 0, 1]])
    if(u,v == (1,0)): return np.array([[0, 0, 0],[0, -1, 1],[0, 0, 0]])
    if(u,v == (1,-1)): return np.array([[1, 0, 0],[0, -1, 0],[0, 0, 0]])
    if(u,v == (-1,-1)): return np.array([[0, 0, 1],[0, -1, 0],[0, 0, 0]])
    if(u,v == (-1,1)): return np.array([[0, 0, 0],[0, -1, 0],[1, 0, 0]])
    if(u,v == (-1,0)): return np.array([[0, 0, 0],[1, -1, 0],[0, 0, 0]])
    if(u,v == (0,1)): return np.array([[0, 0, 0],[0, -1, 0],[0, 1, 0]])
    if(u,v == (0,-1)): return np.array([[0, 1, 0],[0, -1, 0],[0, 0, 0]])


def simple_score(I, w):

    
    score = np.zeros(I.shape)
    for u in range(-1,1):
        for v in range(-1,1):
            imdiff = ndimage.convolve(I,f(u,v),mode='nearest')
            imdiff2 = imdiff ** 2
            score = score + ndimage.gaussian_filter(imdiff2, w)

    
    return score


#--------------------------------------------------------------------------
#                                    Harris score function (Implement this)
#--------------------------------------------------------------------------
def harris_score(I, w):

    fx = np.array([[0, 0, 0],[-1, 0, 1],[0, 0, 0]])
    fy = np.array([[0, -1, 0],[0, 0, 0],[0, 1, 0]])

    dx = ndimage.convolve(I,fx,mode='nearest')
    dy = ndimage.convolve(I,fy,mode='nearest')

    Ixx = dx * dx
    Ixy = dx * dy
    Iyy = dy * dy


    Ixx = ndimage.gaussian_filter(Ixx, w)
    Ixy = ndimage.gaussian_filter(Ixy, w)
    Iyy = ndimage.gaussian_filter(Iyy, w)

    corner_score = np.zeros(I.shape)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            corner_score[i,j] = ((Ixx[i,j] * Iyy[i,j]) - (Ixy[i,j] ** 2)) - (.04* ((Ixx[i,j] + Iyy[i,j]) ** 2))

        
    
    return corner_score
