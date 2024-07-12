import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import *
from scipy import ndimage
from skimage import color
from skimage import io
from skimage.color import rgb2gray

def imageGradient(im):
    
    fx = np.array([[0, 0, 0],[-1, 0, 1],[0, 0, 0]])
    fy = np.array([[0, -1, 0],[0, 0, 0],[0, 1, 0]])


    gx = ndimage.convolve(im, fx, mode='nearest')
    gy = ndimage.convolve(im, fy, mode='nearest')

    img_eps = np.finfo(float).eps


    a1 = (((gx**2)+(gy**2))**(1/2))
    a2 = np.arctan((gy/(gx + img_eps)))

    return a1,a2

def plotit(img):
   
    m,a = imageGradient(img)
  

    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.title('Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(m, cmap=plt.get_cmap('gray'))
    plt.title('Gradient magnitue')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(a, cmap=plt.get_cmap('viridis'))
    plt.colorbar()
    plt.title('Gradient angle')
    plt.axis('off')
    a = a*180/np.pi
    a9 = ((a)+90)*(8/180)
    a9 = (np.rint(a9)).astype(int) 
   
 
    ahist = np.zeros(9, dtype = int)
    h = a9.shape[0]
    w = a9.shape[1]
    for y in range(h):
        for x in range(w):
            i = a9[y,x]
            ahist[i] = ahist[i]+1
            
    plt.subplot(2, 2, 4)
    plt.bar(np.arange(1,10), ahist.tolist())
    plt.title('Gradient histogram')
    plt.xlabel('Bin')
    plt.ylabel('Total magnitude')
    plt.tight_layout()

    
   

if __name__ == '__main__':
    smoothing = False
    im = rgb2gray(utils.imread('../data/parrot.jpg'))
    #im = rgb2gray(utils.imread('../data/butterfly.png'))
    plt.figure(1)
    plotit(im)
    plt.figure(2)
    ims = np.clip(ndimage.gaussian_filter(im, 2),0,1)
    plotit(ims)

    plt.show()




