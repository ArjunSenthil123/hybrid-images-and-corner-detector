import numpy as np
import matplotlib.pyplot as plt 


from scipy.ndimage.filters import convolve, gaussian_filter
from scipy import ndimage, datasets
from vis_hybrid_image import *
import utils
from utils import *
from PIL import Image

def hybrid_image(im1, im2, sigma1, sigma2):


    h = im1.shape[0]
    w = im1.shape[1]
    im1R = im1[:,:,0]
    im1G = im1[:,:,1]
    im1B = im1[:,:,2]

    im2R = im2[:,:,0]
    im2G = im2[:,:,1]
    im2B = im2[:,:,2]

    gim1R = np.clip(ndimage.gaussian_filter(im1R, sigma1),0,1)
    gim1G = np.clip(ndimage.gaussian_filter(im1G, sigma1),0,1)
    gim1B = np.clip(ndimage.gaussian_filter(im1B, sigma1),0,1)

    gim2R = np.clip(ndimage.gaussian_filter(im2R, sigma2),0,1)
    gim2G = np.clip(ndimage.gaussian_filter(im2G, sigma2),0,1)
    gim2B = np.clip(ndimage.gaussian_filter(im2B, sigma2),0,1)

    

    g1 = np.zeros(im1.shape, dtype=float)
    g1[:,:,0] = gim1R
    g1[:,:,1] = gim1G
    g1[:,:,2] = gim1B


    g2 = np.zeros(im2.shape, dtype=float)
    g2[:,:,0] = gim2R
    g2[:,:,1] = gim2G
    g2[:,:,2] = gim2B
    

    himage = g1 + (im2 - g2)
    return himage

if __name__ == '__main__':
    imgname = 'dog.jpg'
    data_dir = os.path.join('..', 'data')
    imgpath = os.path.join(data_dir, imgname)
    im1 = imread(imgpath)

    imgname = 'cat.jpg'
    imgpath = os.path.join(data_dir, imgname)
    im2 = imread(imgpath)
  
    
    himage = hybrid_image(im1,im2,4,10)

    plt.imshow(himage)
    plt.show()

    imgname = 'face1.png'
    imgpath = os.path.join(data_dir, imgname)
    im3 = imread(imgpath)
   

    imgname = 'face2.png'
    imgpath = os.path.join(data_dir, imgname)
    im4 = imread(imgpath)



    himage2 = hybrid_image(im3,im4,4,1)

    plt.imshow(himage2)
    plt.show()
    
    himages = vis_hybrid_image(himage)
    plt.imshow(himages)


    plt.show()
