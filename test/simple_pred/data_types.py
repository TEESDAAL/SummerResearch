import numpy as np, numpy.typing as npt
from typing import Callable
class Img:
    pass

class X:
    pass

class Y:
    pass

class Size:
    pass

class Regions:
    pass

class Vector:
    pass

class Padding:
    pass

class SmallPadding:
    pass

class NumClusters:
    pass

class Scorer:
    pass

class LowerBound:
    pass

class SafeRange:
    pass

class FilteredImg:
    pass

class Weight:
    pass

point = tuple[float, float]
scorer = Callable[[list[point]], float]
image = npt.NDArray[np.floating]
region = tuple[int, int, int, int]
vector = npt.NDArray[np.floating]

