import operator
from functools import partial
import numpy as np
from deap import gp
from data_types import (Img, X, Y, Size,
                        Region, region, Prediction, prediction,
                        )  # defined by author

from parameters import image_width, image_height
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from skimage.feature import hog
import math
import random


def gaussian_1(img_region: region) -> region:
    return ndimage.gaussian_filter(img_region, sigma=1)


#gaussian filter with sigma=1 with the second derivatives
def gaussian_11(img_region: region) -> region:
    return ndimage.gaussian_filter(img_region, sigma=1, order=1)


#gaussian_gradient_magnitude(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gauGM(img_region: region) -> region:
    return ndimage.gaussian_gradient_magnitude(img_region, sigma=1)


#gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gaussian_Laplace1(img_region: region) -> region:
    return ndimage.gaussian_laplace(img_region, sigma=1)


def gaussian_Laplace2(img_region: region) -> region:
    return ndimage.gaussian_laplace(img_region, sigma=2)


#laplace(input, output=None, mode='reflect', cval=0.0)
def laplace(img_region: region) -> region:
    return ndimage.laplace(img_region)


#sobel(input, axis=-1, output=None, mode='reflect', cval=0.0)
def sobel_x(img_region: region) -> region:
    return ndimage.sobel(img_region, axis=0)


def sobel_y(img_region: region) -> region:
    return ndimage.sobel(img_region, axis=1)


def lbp(img_region: region) -> region:
    # 'uniform','default','ror','var'
    return np.divide(
        local_binary_pattern(np.multiply(img_region, 255).astype(int), 8, 1.5, method='nri_uniform'),
        59
    )


def hist_equal(img_region: region) -> region:
    return equalize_hist(img_region, nbins=256, mask=None)


def hog_feature(img_region: region) -> region:
    img, realImage = hog(img_region, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                         transform_sqrt=False, feature_vector=True)
    return realImage


def square_region(img_region: region, x: int, y: int, window_size: int) -> region:
    width, height = img_region.shape
    x_end = min(width, x + window_size)
    y_end = min(height, y + window_size)
    return img_region[x:x_end, y:y_end]


def rect_region(img_region: region, x: int, y: int, width: int, height: int) -> region:
    width, height = img_region.shape
    x_end = min(width, x + width)
    y_end = min(height, y + height)
    return img_region[x:x_end, y:y_end]


def combine(a: float, b: float) -> prediction:
    return complex(a, b)




pset = gp.PrimitiveSetTyped('MAIN', [Img], Prediction)
pset.addPrimitive(combine, [float, float], Prediction, name="combine")

pset.addPrimitive(np.std, [Region], float, name='std')
pset.addPrimitive(np.mean, [Region], float, name='mean')
pset.addPrimitive(np.max, [Region], float, name='max')
pset.addPrimitive(np.min, [Region], float, name='min')

pset.addPrimitive(hist_equal, [Region], Region, name='Hist_Eq')
pset.addPrimitive(gaussian_1, [Region], Region, name='Gau1')
pset.addPrimitive(gaussian_11, [Region], Region, name='Gau11')
pset.addPrimitive(gauGM, [Region], Region, name='GauXY')
pset.addPrimitive(laplace, [Region], Region, name='Lap')
pset.addPrimitive(sobel_x, [Region], Region, name='Sobel_X')
pset.addPrimitive(sobel_y, [Region], Region, name='Sobel_Y')
pset.addPrimitive(gaussian_Laplace1, [Region], Region, name='LoG1')
pset.addPrimitive(gaussian_Laplace2, [Region], Region, name='LoG2')
pset.addPrimitive(lbp, [Region], Region, name='LBP')
pset.addPrimitive(hog_feature, [Region], Region, name='HOG')
# Functions  at the region detection layer
pset.addPrimitive(square_region, [Img, X, Y, Size], Region, name='Region_S')
pset.addPrimitive(rect_region, [Img, X, Y, Size, Size], Region, name='Region_R')

def transform(value: complex, scalar: complex) -> complex:
    return value * scalar

pset.addPrimitive(transform, [Prediction, Prediction], Prediction, name="transform")

def gen_rotator() -> complex:
    theta = random.random() * 2*math.pi
    return complex(math.cos(theta), math.sin(theta))

def gen_scalar() -> complex:
    return complex(6*(random.random() - 0.5), 0)

pset.addEphemeralConstant('scale', gen_scalar, Prediction)
pset.addEphemeralConstant('rotate', gen_rotator, Prediction)

pset.addEphemeralConstant('X', partial(random.randint, 0, image_width - 24), X)
pset.addEphemeralConstant('Y', partial(random.randint, 0, image_height - 24), Y)

# Changed from 20 to 24
pset.addEphemeralConstant('Size', partial(random.randint, 24, 60), Size)

pset.renameArguments(ARG0='Image')
