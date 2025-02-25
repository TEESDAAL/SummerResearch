import numpy as np, numpy.typing as npt
from typing import Callable
class Img:
    pass

class Vector:
    pass

class Std:
    pass

class Order:
    pass

class Orientation:
    pass

class Frequency:
    pass

class Weight:
    pass

class KernelSize:
    pass

point = tuple[float, float]
scorer = Callable[[list[point]], float]
image = npt.NDArray[np.floating]
region = tuple[int, int, int, int]
vector = npt.NDArray[np.floating]


class Vector1:
    pass


class Float3:
    pass


class Img1:
    pass


class Int1:
    pass


class Int2:
    pass


class Int3:
    pass


class Float1:
    pass


class Float2:
    pass