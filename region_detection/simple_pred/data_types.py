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

class WindowSize:
    pass

point = tuple[float, float]
scorer = Callable[[list[point]], float]
image = npt.NDArray[np.floating]
filtered_image = npt.NDArray[np.bool]
region = tuple[int, int, int, int]
vector = npt.NDArray[np.floating]

