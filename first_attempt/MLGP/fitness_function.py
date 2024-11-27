import numpy as np
from deap import gp
import math
from data_types import image
from typing import Callable

def evaluate_valence(individual: gp.PrimitiveTree, compiler: Callable[[gp.PrimitiveTree], Callable], x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
    model = compiler(individual)

    square_errors = [(model(img) - float(val))**2 for img, (val, _) in zip(x_train, y_train)]
    return math.sqrt(sum(square_errors) / len(square_errors)),


def evaluate_arousal(individual: gp.PrimitiveTree, compiler: Callable[[gp.PrimitiveTree], Callable], x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
    model = compiler(individual)

    square_errors = [(model(img) - float(aro))**2 for img, (_, aro) in zip(x_train, y_train)]
    return math.sqrt(sum(square_errors) / len(square_errors)),


def evaluate(model: Callable[[image], tuple[float, float]], x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)
    square_errors = [
        squared_distance(model(x), y) for x, y in zip(x_train, y_train)
    ]

    return sum(square_errors) / len(square_errors),


def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (float(t1[0]) - float(t2[0]))**2 + (float(t1[1]) - float(t2[1]))**2

