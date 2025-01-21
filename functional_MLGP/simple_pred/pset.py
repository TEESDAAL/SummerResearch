import operator, numpy as np, random
from typing import Callable
from deap import gp
from functools import partial
from simple_pred.function_set import (
    hist_equal, gaussian_1, gaussian_11, gauGM, laplace, sobel_x, sobel_y,
    gaussian_Laplace1, gaussian_Laplace2, lbp, hog_feature, safe_div,
    square_region, rect_region, combine, wrapper, binary_wrapper,
    starting_point, threashold, nullary_wrapper
)

from simple_pred.data_types import (
    Img, X, Y, Size, Region, ImageProcessingFunction,
    NumberProducingFunction, PredictionProducingFunction,
    RegionProducingFunction, FilteredImgProducer
)

def create_pset(image_width: int, image_height: int) -> gp.PrimitiveSetTyped:

    pset = gp.PrimitiveSetTyped('MAIN', [], tuple[RegionProducingFunction, PredictionProducingFunction])
    add_image_processing_functions(pset)
    add_regression(pset)
    return pset


def add_image_processing_functions(pset):
    image_processing_layer = [
        (hist_equal, 'Hist_Eq'), (gaussian_1, 'Gau1'), (gaussian_11, 'Gau11'),
        (gauGM, 'GauXY'), (laplace, 'Lap'), (sobel_x, 'Sobel_X'),
        (sobel_y, 'Sobel_Y'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (lbp, 'LBP'), (hog_feature, 'HOG'),
    ]

    for func, name in image_processing_layer:
        pset.addPrimitive(
            partial(wrapper, func), [ImageProcessingFunction],
            ImageProcessingFunction, name=f"{name}_p"
        )


    image_combination = [np.add, np.subtract]

    for func in image_combination:
        pset.addPrimitive(
            partial(binary_wrapper, func), [ImageProcessingFunction]*2,
            ImageProcessingFunction, name=func.__name__
        )

    pset.addPrimitive(starting_point, [], ImageProcessingFunction, name="start")


def add_regression(pset: gp.PrimitiveSetTyped) -> gp.PrimitiveSetTyped:
    # Concatination layer
    pset.addPrimitive(partial(binary_wrapper, combine), [NumberProducingFunction]*2, PredictionProducingFunction, name="combine_p")


    feature_construction_layer = [np.std, np.mean, np.min, np.max]
    for func in feature_construction_layer:
        pset.addPrimitive(partial(wrapper, func), [ImageProcessingFunction], NumberProducingFunction, name=f"{func.__name__}_p")

    binary_operators = [operator.add, operator.sub, operator.mul, safe_div]

    for func in binary_operators:
        pset.addPrimitive(partial(binary_wrapper, func), [NumberProducingFunction]*2, NumberProducingFunction, name=f"{func.__name__}_p")


    return pset


def add_region_detection(pset, image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    pset.addPrimitive(partial(binary_wrapper, threashold), [ImageProcessingFunction, LowerBoundProducer], FilteredImgProducer)
    pset.addEphemeralConstant('lower_bound', random.random, LowerBound)

    pset.addPrimitive(
        partial(centroid_clustering_to_region, clustering_method=KMeans),
        [FilteredImgProducer, NumClusters, Padding, Padding], Regions,
        name="KMeans"
    )

    density_clustering = [AgglomerativeClustering]
    for clustering_method in density_clustering:
        pset.addPrimitive(
            partial(density_clustering_to_region, clustering_method=clustering_method),
            [FilteredImg, NumClusters, SmallPadding, SmallPadding], RegionProducingFunction,
            name=f'{clustering_method.__name__}'
        )

    #pset.addPrimitive(gen_scorer, [Weight]*4, Scorer, name="scorer")

    pset.addEphemeralConstant('s_padding', partial(random.randint, -5, 5), SmallPadding)
    pset.addEphemeralConstant('l_padding', partial(random.randint, image_width // 20, image_width // 5), Padding)
    #pset.addEphemeralConstant('l_padding', partial(random.randint, 5, 12), Padding)

    #pset.addEphemeralConstant('clusters', partial(random.randint, 1, 7), NumClusters)
    pset.addTerminal(3, NumClusters)
    # Changed from 20 to 24
    pset.addEphemeralConstant('size', partial(random.randint, 24, 60), Size)
    pset.addEphemeralConstant('weight', gen_weight, Weight)
    pset.renameArguments(ARG0='Image')

    return pset



