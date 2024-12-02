from functools import partial
import numpy as np
from deap import gp
from complex_num_pred.data_types import (Img, X, Y, Size,
                        Region, region, Prediction, prediction,
                        )  # defined by author

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


def gen_rotator() -> complex:
        theta = random.random() * 2*math.pi
        return complex(math.cos(theta), math.sin(theta))


def gen_scalar() -> complex:
    return complex(6*(random.random() - 0.5), 0)


def transform(value: complex, scalar: complex) -> complex:
        return value * scalar


def create_pset(image_width, image_height) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Prediction)
    pset.addPrimitive(combine, [float, float], Prediction, name="combine")

    feature_construction_layer = [np.std, np.mean, np.min, np.max]
    for func in feature_construction_layer:
        pset.addPrimitive(func, [Region], float, name=func.__name__)

    image_processing_layer = [
        (hist_equal, 'Hist_Eq'), (gaussian_1, 'Gau1'), (gaussian_11, 'Gau11'),
        (gauGM, 'GauXY'), (laplace, 'Lap'), (sobel_x, 'Sobel_X'),
        (sobel_y, 'Sobel_Y'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (lbp, 'LBP'), (hog_feature, 'HOG'),
    ]

    for func, name in image_processing_layer:
        pset.addPrimitive(func, [Region], Region, name=name)



    # Functions  at the region detection layer
    pset.addPrimitive(square_region, [Img, X, Y, Size], Region, name='Region_S')
    pset.addPrimitive(rect_region, [Img, X, Y, Size, Size], Region, name='Region_R')


    pset.addPrimitive(transform, [Prediction, Prediction], Prediction, name="transform")


    pset.addEphemeralConstant('scale', gen_scalar, Prediction)
    pset.addEphemeralConstant('rotate', gen_rotator, Prediction)

    pset.addEphemeralConstant('X', partial(random.randint, 0, image_width - 24), X)
    pset.addEphemeralConstant('Y', partial(random.randint, 0, image_height - 24), Y)

    # Changed from 20 to 24
    pset.addEphemeralConstant('Size', partial(random.randint, 24, 60), Size)

    pset.renameArguments(ARG0='Image')

    return pset
