from deap import gp
from typing import Callable
import numpy as np
from shared_tools.make_datasets import x_train

x = y = width = height = int
Region = tuple[x, y, width, height]
Point = tuple[x, y]
Landmarks = list[Point]


def evaluate(individual: gp.PrimitiveTree, compiler: Callable[[gp.PrimitiveTree], Callable], xs: np.ndarray, ys: np.ndarray) -> tuple[float]:
    regions_extractor = compiler(individual)
    errors: list[float] = [
        error(regions_extractor(image), landmarks) for image, landmarks in zip(xs, ys)
    ]
    return sum(errors) / len(errors),


def error(regions, points) -> float:
    captured_points = sum(int(inside_regions(p, regions)) for p in points)
    area = sum(region[2] * region[3] for region in regions)
    if area == 0:
        return 0
    return captured_points / (len(points) * np.sqrt(area))


def inside_regions(point: Point, regions: list[Region]) -> bool:
    return any(inside_region(point, region) for region in regions)


def inside_region(point: Point, region: Region) -> bool:
    region_x, region_y, width, height = region
    x, y = point
    return region_x <= x <= region_x + width and  region_y <= y <= region_y + height

