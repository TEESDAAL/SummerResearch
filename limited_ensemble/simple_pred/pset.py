import operator, numpy as np, random
from typing import Callable
from deap import gp
from functools import partial
from simple_pred.function_set import (
    hist_equal, gaussian_1, gaussian_11, gauGM, laplace, sobel_x, sobel_y,
    gaussian_Laplace1, gaussian_Laplace2, lbp, hog_feature,
    square_region, rect_region, to_array
)

from simple_pred.data_types import (
    Ensemble, Img, X, Y, Size, Region, Region1, Region2, Region3,
)


def create_pset(image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Ensemble)

    for ensemble_size in [3, 5, 7, 10, 15]:
        pset.addPrimitive(to_array, [float]*ensemble_size, Ensemble, name=f"combine{ensemble_size}")


    feature_construction_layer = [np.std, np.mean]
    for func in feature_construction_layer:
        pset.addPrimitive(func, [Region], float, name=func.__name__)
        pset.addPrimitive(func, [Region1], float, name=func.__name__)
        pset.addPrimitive(func, [Region2], float, name=func.__name__)
        pset.addPrimitive(func, [Region3], float, name=func.__name__)

    image_processing_layer = [
        (hist_equal, 'Hist_Eq'), (gaussian_1, 'Gau1'), (gaussian_11, 'Gau11'),
        (gauGM, 'GauXY'), (laplace, 'Lap'), (sobel_x, 'Sobel_X'),
        (sobel_y, 'Sobel_Y'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (lbp, 'LBP'), (hog_feature, 'HOG'),
    ]

    for func, name in image_processing_layer:
        pset.addPrimitive(func, [Region], Region1, name=name)
        pset.addPrimitive(func, [Region1], Region2, name=name)
        pset.addPrimitive(func, [Region2], Region3, name=name)



    # Functions  at the region detection layer
    pset.addPrimitive(square_region, [Img, X, Y, Size], Region, name='Region_S')
    pset.addPrimitive(rect_region, [Img, X, Y, Size, Size], Region, name='Region_R')


    pset.addEphemeralConstant('X', partial(random.randint, 0, image_width - 24), X)
    pset.addEphemeralConstant('Y', partial(random.randint, 0, image_height - 24), Y)
    # Changed from 20 to 24
    pset.addEphemeralConstant('Size', partial(random.randint, 24, 60), Size)

    pset.renameArguments(ARG0='Image')

    return pset


def region_in(pset, func: Callable, ret_type: type, name:str):
    pset.addPrimitive(func, [Region], ret_type, name=name)
    pset.addPrimitive(func, [Region1], ret_type, name=name)
    pset.addPrimitive(func, [Region2], ret_type, name=name)
    pset.addPrimitive(func, [Region2], ret_type, name=name)




