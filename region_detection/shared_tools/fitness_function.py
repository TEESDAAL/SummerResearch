from deap import gp, base
from typing import Callable
import numpy as np
from shared_tools.make_datasets import x_train
from functools import partial

x = y = width = height = int
Region = tuple[x, y, width, height]
Point = tuple[x, y]
Landmarks = list[Point]
cache = {}

def evaluate(individual: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray, mode: str) -> tuple[float]:
    key = str(individual) + mode
    if key in cache:
        toolbox.cache_hits.value += 1
        return cache[key],

    regions = toolbox.parallel_map(
        partial(extract_regions, individual=individual, compiler=toolbox.compile),
        xs
    )

    errors: list[float] = list(toolbox.parallel_map(error, zip(regions, ys)))

    fitness = sum(errors) / len(errors)
    cache[key] = fitness
    return fitness,


def extract_regions(image, individual, compiler) -> list[Region]:
    return compiler(individual)(image)


def error(regions_points) -> float:
    regions, points = regions_points
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

