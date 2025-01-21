from typing import Callable, Any, TypeVar
import numpy as np

A, B, T = TypeVar('A'), TypeVar('B'), TypeVar('T')


class Img:
    pass

class X:
    pass

class Y:
    pass

class Size:
    pass

class Region:
    pass


class BinaryFunction:
    pass

class Prediction:
    pass

class Double:
    pass


class PredictionPair:
    pass

class NumberProducingFunction:
    pass

class ImageProcessingFunction:
    pass

class PredictionProducingFunction:
    pass

class RegionProducingFunction:
    pass

class FilteredImgProducer:
    pass

image = np.ndarray
region = np.ndarray
prediction = tuple[float, float]
binary_function = Callable[[float, float], float]
prediction_pair = tuple[prediction, prediction]
unary_function = Callable[[float], float]
scalar = float
image_processing_function = Callable[[image], T]
