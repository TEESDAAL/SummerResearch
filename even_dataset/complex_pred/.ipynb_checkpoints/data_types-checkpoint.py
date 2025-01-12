from typing import Callable

import numpy as np

class Img:
    pass

class X:
    pass

class Y:
    pass

class Size:
    pass

# class Region:
#     pass
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

class UnaryFunction:
    pass

class Scalar:
    pass

image = np.ndarray
region = np.ndarray
prediction = tuple[float, float]
binary_function = Callable[[float, float], float]
prediction_pair = tuple[prediction, prediction]
unary_function = Callable[[float], float]
scalar = float

