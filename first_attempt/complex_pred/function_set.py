import operator
from functools import partial
import numpy as np
from deap import gp
from data_types import (Img, X, Y, Size,
                        Region, region, Prediction, prediction,
                        PredictionPair, prediction_pair, BinaryFunction, binary_function,
                        UnaryFunction, unary_function, Scalar, scalar
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
    return a, b


def split(pred: prediction) -> prediction_pair:
    return pred, pred


pset = gp.PrimitiveSetTyped('MAIN', [Img], Prediction)
pset.addPrimitive(combine, [float, float], Prediction, name="combine")
pset.addPrimitive(split, [Prediction], PredictionPair, name="split")

def complex_combine(ps: prediction_pair, left_combiner: binary_function,
                    right_combiner: binary_function) -> prediction:
    (l1, r1), (l2, r2) = ps
    return left_combiner(l1, l2), right_combiner(r1, r2)

pset.addPrimitive(complex_combine, [PredictionPair, BinaryFunction, BinaryFunction], Prediction, name='complex_combine')


pset.addPrimitive(np.std, [Region], float, name='G_std')
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



def safe_div(a, b):
    if b == 0:
        return 1
    return a / b

def left(a, _):
    return a

def right(_, b):
    return b

for bi_function in [operator.add, operator.sub, operator.mul, safe_div, left, right]:
    pset.addTerminal(bi_function, BinaryFunction, name=bi_function.__name__)




def left_left_apply(pred_pair: prediction_pair, f: unary_function) -> prediction_pair:
    (l1, l2), right = pred_pair

    return (f(l1), l2), right

def left_right_apply(pred_pair: prediction_pair, f: unary_function) -> prediction_pair:
    (l1, l2), right = pred_pair

    return (l1, f(l2)), right

def right_left_apply(pred_pair: prediction_pair, f: unary_function) -> prediction_pair:
    left, (r1, r2) = pred_pair
    return left, (f(r1), r2)

def right_right_apply(pred_pair: prediction_pair, f: unary_function) -> prediction_pair:
    left, (r1, r2) = pred_pair
    return left, (r1, f(r2))

for applicator in [left_left_apply, left_right_apply, right_left_apply, right_right_apply]:
    pset.addPrimitive(applicator, [PredictionPair, UnaryFunction], PredictionPair, name=applicator.__name__)

def genScalar() -> scalar:
    return float(random.choice(np.arange(-2, 2.1, 0.25)))

pset.addEphemeralConstant('randScalar', genScalar, Scalar)
def safe_log(value):
    if value == 0:
        value += 0.001

    return math.log(abs(value))
for unary_func in [math.sin, math.cos, safe_log]:
    pset.addTerminal(unary_func, UnaryFunction, name=unary_func.__name__)

def curried_scale(value: scalar) -> unary_function:
    return partial(scale, value)

def scale(value: float, scalar: float) -> float:
    return value*scalar

pset.addPrimitive(curried_scale, [Scalar], UnaryFunction, name="scalar")
pset.addEphemeralConstant('X', partial(random.randint, 0, image_width - 24), X)
pset.addEphemeralConstant('Y', partial(random.randint, 0, image_height - 24), Y)

# Changed from 20 to 24
pset.addEphemeralConstant('Size', partial(random.randint, 24, 60), Size)

pset.renameArguments(ARG0='Image')
