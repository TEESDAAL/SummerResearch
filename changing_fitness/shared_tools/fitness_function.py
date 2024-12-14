from deap import gp
from typing import Callable
import numpy as np


def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(individual: gp.PrimitiveTree, compiler: Callable[[gp.PrimitiveTree], Callable], x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    model = compiler(individual)

    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)
    square_errors = [
       squared_distance(model(x), y) for x, y in zip(x_train, y_train)
    ]

    return sum(square_errors) / len(square_errors),

def error(x_y_w: tuple[np.ndarray, tuple[float, float], float], individual, compiler: Callable[[gp.PrimitiveTree], Callable]):
    model = compiler(individual)
    x, y, w = x_y_w
    return w*squared_distance(model(x), y)
