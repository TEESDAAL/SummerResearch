from deap import gp, base
from functools import partial
import numpy as np

cache = {}


def evaluate(individual: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray, mode: str) -> tuple[float]:
    """
    Evaluate a given model for it's erorr for a given input.
    This is cached based on the string representation of the individual and the mode.
    The idea being if two models have the same string representation,
        and are being evaluated on the same data, they should have the same error

    Parameters
    ----------
    individual : deap.gp.PrimitiveTree
        The GP model to evaluate.
    toolbox : deap.base.Toolbox
        The toolbox that holds the tools to evaluate a model
    xs : np.ndarray
        A numpy array of images
    ys : np.ndarray
        A numpy array the appropriate labels for the xs
    mode: str
        A string to make the caching work, should be unique to the (xs, ys) pair. Usual values are "train" and "val"
    Returns
    -------
    tuple[float]
        A one element tuple representing the error (squared distance of the true value and the predicted value)
    """
    key = str(individual) + mode
    if key in cache:
        return cache[key],

    square_errors = list(toolbox.parallel_map(partial(error, individual=individual, compiler=toolbox.compile), zip(xs, ys)))
    cache[key] = mean_squared_error = sum(square_errors) / len(square_errors)

    return mean_squared_error,

def error(x_y, individual, compiler):
    x, y = x_y
    return squared_distance(compiler(individual)(x), y)

def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2
