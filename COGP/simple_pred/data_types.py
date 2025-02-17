import numpy as np, numpy.typing as npt
from typing import Callable
class Img:
    pass

class Vector:
    pass

class ConvolvedImg:
    pass

class ConvolvedPooledImg:
    pass

class Weight:
    pass

class KernelSize:
    pass

class Filter:
    pass


point = tuple[float, float]
scorer = Callable[[list[point]], float]
image = npt.NDArray[np.floating]
region = tuple[int, int, int, int]
vector = npt.NDArray[np.floating]

