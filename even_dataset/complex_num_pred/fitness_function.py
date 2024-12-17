from typing import Callable
from complex_num_pred.data_types import prediction
from deap import gp
import numpy as np


def squared_distance(c1: prediction, t: tuple[float, float]) -> float:
    distance = c1 - complex(*t)
    return distance.real**2 + distance.imag**2


def evaluate(individual: gp.PrimitiveTree, compiler: Callable[[gp.PrimitiveTree], Callable], x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
    """Compute the MSE  between the models answer and the true (val, aro) pair. Uses square distances avoids computing the sqrt"""
    model = compiler(individual)
    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)
    square_errors = [
        squared_distance(model(x), y) for x, y in zip(x_train, y_train)
    ]

    return sum(square_errors) / len(square_errors),

def error(x_y: tuple[np.ndarray, tuple[float, float]], individual, compiler: Callable[[gp.PrimitiveTree], Callable]):
    model = compiler(individual)
    return squared_distance(model(x_y[0]), x_y[1])
