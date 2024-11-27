import numpy as np
from deap import gp
from data_types import Img, X, Y, Size, Region, Double  # defined by author
from parameters import image_width, image_height
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from skimage.feature import hog
import random


def gaussian_1(left):
    return ndimage.gaussian_filter(left,sigma=1)

#gaussian filter with sigma=1 with the second derivatives
def gaussian_11(left):
    return ndimage.gaussian_filter(left, sigma=1,order=1)

#gaussian_gradient_magnitude(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gauGM(left):
    return ndimage.gaussian_gradient_magnitude(left,sigma=1)

#gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gaussian_Laplace1(left):
    return ndimage.gaussian_laplace(left,sigma=1)

def gaussian_Laplace2(left):
    return ndimage.gaussian_laplace(left,sigma=2)

#laplace(input, output=None, mode='reflect', cval=0.0)
def laplace(left):
    return ndimage.laplace(left)

#sobel(input, axis=-1, output=None, mode='reflect', cval=0.0)
def sobelx(left):
    return ndimage.sobel(left,axis=0)

def sobely(left):
    return ndimage.sobel(left,axis=1)

def lbp(image):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(np.multiply(image, 255).astype(int), 8, 1.5, method='nri_uniform')
    lbp=np.divide(lbp,59)

    return lbp

def hist_equal(image):
    return equalize_hist(image, nbins=256, mask=None)

def hog_feature(image):
    img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                         transform_sqrt=False, feature_vector=True)
    return realImage

def regionS(left,x,y,windowSize):
    width,height=left.shape
    x_end = min(width, x+windowSize)
    y_end = min(height, y+windowSize)
    slice = left[x:x_end, y:y_end]
    return slice

def regionR(left, x, y, windowSize1,windowSize2):
    width, height = left.shape
    x_end = min(width, x + windowSize1)
    y_end = min(height, y + windowSize2)
    slice = left[x:x_end, y:y_end]
    return slice

pset = gp.PrimitiveSetTyped('MAIN', [Img], Double)

pset.addPrimitive(np.std, [Region], Double, name='std')
pset.addPrimitive(np.mean, [Region], Double, name='mean')
pset.addPrimitive(np.min, [Region], Double, name='min')
pset.addPrimitive(np.max, [Region], Double, name='max')

pset.addPrimitive(hist_equal, [Region], Region, name='Hist_Eq')
pset.addPrimitive(gaussian_1, [Region], Region, name='Gau1')
pset.addPrimitive(gaussian_11, [Region], Region, name='Gau11')
pset.addPrimitive(gauGM, [Region], Region, name='GauXY')
pset.addPrimitive(laplace, [Region], Region, name='Lap')
pset.addPrimitive(sobelx, [Region], Region, name='Sobel_X')
pset.addPrimitive(sobely, [Region], Region, name='Sobel_Y')
pset.addPrimitive(gaussian_Laplace1, [Region], Region, name='LoG1')
pset.addPrimitive(gaussian_Laplace2, [Region], Region, name='LoG2')
pset.addPrimitive(lbp, [Region], Region, name='LBP')
pset.addPrimitive(hog_feature, [Region], Region, name='HOG')
# Functions  at the region detection layer
pset.addPrimitive(regionS, [Img, X, Y, Size], Region, name='Region_S')
pset.addPrimitive(regionR, [Img, X, Y, Size, Size], Region, name='Region_R')
pset.renameArguments(ARG0='Image')
# WHY IS THIS HERE?
#pset.addEphemeralConstant('randomDouble', lambda: round(random.random(), 2), float)
def genX():
    return random.randint(0, image_width - 24)

def genY():
    return random.randint(0, image_height - 24)

def genSize():
    return random.randint(24, 70)

# pset.addEphemeralConstant('X', lambda: random.randint(0, image_width - 24), Int1)
pset.addEphemeralConstant('X', genX, X)
pset.addEphemeralConstant('Y', genY, Y)

#pset.addEphemeralConstant('Y', lambda: random.randint(0, image_height - 24), Int2)

# Changed from 20 to 24
# pset.addEphemeralConstant('Size', lambda: random.randint(24, 70), Int3)
pset.addEphemeralConstant('Size', genSize, Size)
