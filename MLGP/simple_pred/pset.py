import operator, numpy as np, random
from deap import gp
from functools import partial
from simple_pred.function_set import (
    hist_equal, gaussian_1, gaussian_11, gauGM, laplace, sobel_x, sobel_y,
    gaussian_Laplace1, gaussian_Laplace2, lbp, hog_feature, safe_div,
    square_region, rect_region, combine
)

from simple_pred.data_types import Img, X, Y, Size, Region, Prediction


def create_pset(image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Prediction)

    # Concatination layer
    pset.addPrimitive(combine, [float, float], Prediction, name="combine")


    feature_construction_layer = [np.std, np.mean, np.min, np.max]
    for func in feature_construction_layer:
        pset.addPrimitive(func, [Region], float, name=func.__name__)

    binary_operators = [operator.add, operator.sub, operator.mul, safe_div]

    for func in binary_operators:
        pset.addPrimitive(func, [float, float], float)

    pset.addPrimitive(partial(operator.mul, -1), [float], float, name="neg")
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


    pset.addEphemeralConstant('X', partial(random.randint, 0, image_width - 24), X)
    pset.addEphemeralConstant('Y', partial(random.randint, 0, image_height - 24), Y)

    # Changed from 20 to 24
    pset.addEphemeralConstant('Size', partial(random.randint, 24, 60), Size)

    pset.renameArguments(ARG0='Image')

    return pset
