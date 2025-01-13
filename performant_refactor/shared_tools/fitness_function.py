from deap import gp, base
from typing import Callable
import numpy as np
from functools import partial

cache={}

def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(individual: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray, mode:str) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    key = str(individual)+mode
    if key in cache:
        toolbox.cache_hits.value += 1
        return cache[key],

    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)

    square_errors = list(toolbox.parallel_map(
        partial(error, individual=individual, compiler=toolbox.compile),
        zip(xs, ys)
    ))

    mean_squared_error = sum(square_errors) / len(square_errors)
    cache[key] = mean_squared_error
                                              
    return mean_squared_error,

def error(x_y: tuple[np.ndarray, tuple[float, float]], individual, compiler: Callable[[gp.PrimitiveTree], Callable]):
    model = compiler(individual)
    return squared_distance(model(x_y[0]), x_y[1])
