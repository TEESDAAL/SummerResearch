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

type region = np.ndarray
type prediction = tuple[float, float]
type binary_function = Callable[[float, float], float]
type prediction_pair = tuple[prediction, prediction]
type unary_function = Callable[[float], float]
type scalar = float

