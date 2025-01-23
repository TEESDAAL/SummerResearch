from deap import gp, base
from functools import partial
import numpy as np


def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(individual: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray, mode: str) -> tuple[float]:
    print(individual)
    region_generator, region_consumer = toolbox.compile(individual)()
    print(region_generator, region_consumer)
    return 4,
    square_errors = list(toolbox.parallel_map(partial(error, individual=individual, compiler=toolbox.compile), zip(xs, ys)))
    cache[key] = mean_squared_error = sum(square_errors) / len(square_errors)

    return mean_squared_error,

def error(x_y, individual, compiler):
    model = compiler(individual)()
    x, y = x_y
    return squared_distance(model(x), y)

