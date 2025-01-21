import numpy as np
from typing import Callable
from simple_pred.data_types import (
    image, image, prediction, image_processing_function, A, B, T
)
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from skimage.feature import hog

def gaussian_1(image: image) -> image:
    return ndimage.gaussian_filter(image, sigma=1)


#gaussian filter with sigma=1 with the second derivatives
def gaussian_11(image: image) -> image:
    return ndimage.gaussian_filter(image, sigma=1, order=1)


#gaussian_gradient_magnitude(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gauGM(image: image) -> image:
    return ndimage.gaussian_gradient_magnitude(image, sigma=1)


#gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gaussian_Laplace1(image: image) -> image:
    return ndimage.gaussian_laplace(image, sigma=1)


def gaussian_Laplace2(image: image) -> image:
    return ndimage.gaussian_laplace(image, sigma=2)


#laplace(input, output=None, mode='reflect', cval=0.0)
def laplace(image: image) -> image:
    return ndimage.laplace(image)


#sobel(input, axis=-1, output=None, mode='reflect', cval=0.0)
def sobel_x(image: image) -> image:
    return ndimage.sobel(image, axis=0)


def sobel_y(image: image) -> image:
    return ndimage.sobel(image, axis=1)


def lbp(image: image) -> image:
    # 'uniform','default','ror','var'
    # Convert from float to int to avoid warning about floating point errors
    return np.divide(
        local_binary_pattern(np.multiply(image, 255).astype(int), 8, 1.5, method='nri_uniform'),
        59
    )


def hist_equal(image: image) -> image:
    return equalize_hist(image, nbins=256, mask=None)


def hog_feature(image: image) -> image:
    _, visualized_hog = hog(image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                         transform_sqrt=False, feature_vector=True)
    return visualized_hog


def square_region(image: image, x: int, y: int, window_size: int) -> image:
    return rect_region(image, x, y, window_size, window_size)


def rect_region(image: image, x: int, y: int, width: int, height: int) -> image:
    width, height = image.shape
    x_end = min(width, x + width)
    y_end = min(height, y + height)
    return image[y:y_end, x:x_end]


def combine(a: float, b: float) -> prediction:
    return a, b


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 1

    return a / b

def wrapper(image_processing_func: image_processing_function, prev_function: image_processing_function) -> image_processing_function:
    return lambda image: image_processing_func(prev_function(image))

def binary_wrapper(func: Callable[[A, B], T], arg_func1: image_processing_function[A], arg_func2: image_processing_function[B]) -> image_processing_function[T]:
    return lambda image: func(arg_func1(image), arg_func2(image))
# def nullary_wrapper(func: Callable[[], T]) ->
def starting_point() -> image_processing_function:
    return lambda image: image
