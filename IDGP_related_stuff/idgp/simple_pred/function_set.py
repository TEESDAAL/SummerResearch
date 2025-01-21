from functools import partial
import numpy as np
from deap import gp
from simple_pred.data_types import (Img, image, X, Y, Size, Region, region, Vector, vector)  # defined by author
import random
import simple_pred.sift_features as sift_features
from typing import Union
from skimage.feature import local_binary_pattern, hog

def mean_std(region: Union[image, region]) -> tuple[np.floating, np.floating]:
    return np.mean(region), np.std(region)


def feature_DIF(image: Union[image, region]) -> vector:
    width, height = image.shape
    half_width = width // 2
    half_height = height // 2
    quarter_width = width // 4
    quarter_height = height // 4
    three_quarter_width = half_width + quarter_width
    three_quarter_height = half_height + quarter_height

    result = np.array([
        *mean_std(image),
        *mean_std(image[0:half_width, 0:half_height]),
        *mean_std(image[0:half_width, half_height:height]),
        *mean_std(image[half_width:width, 0:half_height]),
        *mean_std(image[half_width:width, half_height:height]),
        *mean_std(image[quarter_width:three_quarter_width, quarter_height:three_quarter_height]),
        *mean_std(image[half_width, :]),
        *mean_std(image[:, half_height]),
        *mean_std(image[half_width, quarter_height:three_quarter_height]),
        *mean_std(image[quarter_width:three_quarter_width, half_height])
    ])
    assert result.shape == (20,)
    return result


def all_histogram(image: Union[image, region]) -> vector:
    # global and local
    n_bins = 32
    hist, ax = np.histogram(image, n_bins, (0, 1))
    # dimension 24 for all type images
    return hist


def all_sift(image: Union[image, region]) -> vector:
    # global and local
    width, height = image.shape
    min_length = np.min((width,height))
    img = np.asarray(image[0:width,0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length,0:min_length])
    # dimension 128 for all images
    w, h = feaArrSingle.shape
    feature_vector = np.reshape(feaArrSingle, (h,))
    return feature_vector


def all_lbp(image: Union[image, region]) -> vector:
    # global and local
    feature_vector = histLBP(image, 1.5, 8)
    # dimension 59 for all images
    return feature_vector


def LBP(image: Union[image, region], radius: float, n_points: int, method: str='nri_uniform'):
    # 'uniform','default','ror','var'
    return local_binary_pattern(image, n_points, radius, method)


def histLBP(image: Union[image, region], radius: float, n_points: int):
    #uniform_LBP
    lbp = LBP(image, radius=radius, n_points=n_points)
    n_bins = 59
    hist, ax = np.histogram(lbp,n_bins, (0, n_bins))
    return hist

def square_region(img_region: image, x: int, y: int, window_size: int) -> region:
    width, height = img_region.shape
    x_end = min(width, x + window_size)
    y_end = min(height, y + window_size)
    return img_region[x:x_end, y:y_end]


def rect_region(img_region: image, x: int, y: int, width: int, height: int) -> region:
    width, height = img_region.shape
    x_end = min(width, x + width)
    y_end = min(height, y + height)
    return img_region[x:x_end, y:y_end]


def concatenate(*args: vector) -> vector:
    return np.concatenate((args), axis=0)

def global_hog(image):
    feature_vector = hog_features(image, 20, 10)
    # dimension 144 for 128*128
    return feature_vector

def local_hog(image):
    try:
        feature_vector = hog_features(image,10,10)
    except: feature_vector = np.concatenate(image)
    #dimension don't know
    return feature_vector

def hog_features(image: Union[image, region], patch_size: int, moving_size: int) -> vector:
    img = np.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(w):
        for j in range(h):
            patch.append([moving_size * i, moving_size * j])

    hog_features = np.zeros((len(patch)))
    realImage = HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = np.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)]
        )

    return hog_features


def HoGFeatures(image: Union[image, region]) -> image:
    img, realImage = hog(image,orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                transform_sqrt=False, feature_vector=True)
    return realImage



def create_pset(image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector)

    #Feature concatenation CHANGE FROM BOOK CODE - REMOVED REFERENCE TO VECTOR1
    pset.addPrimitive(concatenate, [Vector, Vector], Vector, name='concat2')
    pset.addPrimitive(concatenate, [Vector, Vector, Vector], Vector, name='concat3')
    feature_extraction_functions = [
        (feature_DIF, 'DIF'), (all_histogram, 'Histogram'),
        (all_lbp, 'uLBP'), (all_sift, 'SIFT')
    ]
    for function, name in feature_extraction_functions:
        pset.addPrimitive(function, [Img], Vector, name=f"Global_{name}")
        pset.addPrimitive(function, [Region], Vector, name=f"Local_{name}")

    pset.addPrimitive(global_hog, [Img], Vector, name='Global_HOG')
    pset.addPrimitive(local_hog, [Region], Vector, name='Local_HOG')

    # Region detection operators
    pset.addPrimitive(square_region, [Img, X, Y, Size], Region, name='Region_S')
    pset.addPrimitive(rect_region, [Img, X, Y, Size, Size], Region, name='Region_R')


    pset.addEphemeralConstant('X', partial(random.randint, 0, image_width - 24), X)
    pset.addEphemeralConstant('Y', partial(random.randint, 0, image_height - 24), Y)

    # Differs from book code [20, 70]
    pset.addEphemeralConstant('Size', partial(random.randint, 24, 60), Size)

    pset.renameArguments(ARG0='Image')

    return pset
