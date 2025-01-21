from deap import gp, base
from functools import partial
import numpy as np

cache = {}

def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(individual: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray, mode: str) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    key = str(individual) + mode
    if key in cache:
        return cache[key],

    square_errors = list(toolbox.parallel_map(partial(error, individual=individual, compiler=toolbox.compile), zip(xs, ys)))
    cache[key] = mean_squared_error = sum(square_errors) / len(square_errors)

    return mean_squared_error,

def error(x_y, individual, compiler):
    model = compiler(individual)()
    x, y = x_y
    return squared_distance(model(x), y)

