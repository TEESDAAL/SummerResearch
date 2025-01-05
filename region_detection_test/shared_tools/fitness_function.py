from deap import gp, base
from typing import Callable
import numpy as np
from shared_tools.make_datasets import x_train
from functools import partial

x = y = width = height = int
Region = tuple[x, y, width, height]
Point = tuple[x, y]
Landmarks = list[Point]


def evaluate(individual: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray) -> tuple[float]:
    errors = list(toolbox.parrallel_map(
        partial(evaluate_instance, individual=individual, compiler=toolbox.compile),
        zip(xs, ys)
    ))
    return sum(errors) / len(errors),

def evaluate_instance(x_y, individual, compiler):
    model = compiler(individual)
    return error(model(x_y[0]), x_y[1])

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

