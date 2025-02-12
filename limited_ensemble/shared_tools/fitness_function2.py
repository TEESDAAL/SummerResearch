from deap import gp, base
from functools import partial
import numpy as np


def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(ensemble: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""

    square_errors = list(toolbox.parallel_map(partial(error, ensemble=ensemble, compiler=toolbox.compile), zip(xs, ys)))
    return sum(square_errors) / len(square_errors),

def error(x_y, ensemble, compiler):
    x, y = x_y
    mean_guess = compiler(ensemble)(x)

    return squared_distance(mean_guess, y)

