import numpy as np, random, operator
from deap import gp
from functools import partial
from simple_pred.data_types import (
    Img, image, FilteredImg, Regions, region, LowerBound, NumClusters, Padding, Prediction, Region,
)
from simple_pred.function_set import (
    hist_equal, gaussian_1, gaussian_11, gauGM, laplace, sobel_x, sobel_y,
    lbp, hog_feature, gaussian_Laplace1, gaussian_Laplace2, threashold,
    centroid_clustering_to_region, square_region, rect_region
)
from sklearn.cluster import KMeans
from typing import Callable

region_generator = Callable[[image, int], list[region]]
region_consumer = Callable[[list[region]], float]


def pset(image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped('MAIN', [], tuple[region_generator, region_consumer])

    # pset.renameArguments(ARG0='Regions', ARG1='NumClusters', ARG2='Image')


    pset.addPrimitive(tuple, [region_generator, region_consumer], tuple[region_generator, region_consumer], name="combine")

    add_region_detection(pset, image_width, image_height)
    add_regression_part(pset)

    return pset


def add_region_detection(pset, image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    image_combination = [np.add, np.subtract]
    for func in image_combination:
        pset.addPrimitive(func, [Img, Img], Img, name=func.__name__)


    image_processing_layer = [
        (hist_equal, 'Hist_Eq'), (gaussian_1, 'Gau1'), (gaussian_11, 'Gau11'),
        (gauGM, 'GauXY'), (laplace, 'Lap'), (sobel_x, 'Sobel_X'),
        (sobel_y, 'Sobel_Y'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (abs, 'abs')
    ]

    for func, name in image_processing_layer:
        pset.addPrimitive(func, [Img], Img, name=name)

    pset.addPrimitive(threashold, [Img, LowerBound], FilteredImg)
    pset.addEphemeralConstant('lower_bound', random.random, LowerBound)

    # image: image, number_clusters: int, x_pad: int, y_pad: int, clustering_method
    pset.addPrimitive(
        partial(centroid_clustering_to_region, clustering_method=KMeans),
        [FilteredImg, Padding, Padding], region_generator,
        name="KMeans"
    )

    img_size = min(image_width, image_height)
    pset.addEphemeralConstant('l_padding', partial(random.randint, img_size // 20, img_size // 5), Padding)
    #pset.addEphemeralConstant('l_padding', partial(random.randint, 5, 12), Padding)


    return pset

def select_region(regions: list[region], i: int) -> region:
    return regions[i]

def add_regression_part(pset: gp.PrimitiveSetTyped) -> gp.PrimitiveSetTyped:
    pset.addPrimitive(?, [], region_consumer, name="regression_half")
    pset.addPrimitive(tuple, [float, float], Prediction, name="combine")

    feature_construction_layer = [np.std, np.mean, np.min, np.max]
    for func in feature_construction_layer:
        pset.addPrimitive(func, [Region], float, name=func.__name__)

    binary_operators = [operator.add, operator.sub]

    for func in binary_operators:
        pset.addPrimitive(func, [float, float], float)

    image_processing_layer = [
        (hist_equal, 'Hist_Eq'), (gaussian_1, 'Gau1'), (gaussian_11, 'Gau11'),
        (gauGM, 'GauXY'), (laplace, 'Lap'), (sobel_x, 'Sobel_X'),
        (sobel_y, 'Sobel_Y'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (lbp, 'LBP'), (hog_feature, 'HOG'),
    ]

    for func, name in image_processing_layer:
        pset.addPrimitive(func, [Region], Region, name=name)

    pset.addPrimitive(select_region, [Regions, NumClusters], Region, name="select_region")


    pset.renameArguments(ARG0='Image')

    return pset

def lbp_p(prev_function):
    return lambda image: lbp(prev_function(image))
def test(tree) -> region_consumer:
    return lambda regions: tree(regions)
