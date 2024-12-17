from deap import gp
from typing import Callable
import numpy as np


def evaluate(individual: gp.PrimitiveTree, compiler: Callable[[gp.PrimitiveTree], Callable], x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    errors = [
       error((x, y), individual, compiler) for x, y in zip(x_train, y_train)
    ]

    return sum(errors) / len(errors),

def error(x_y: tuple[np.ndarray, int], individual, compiler: Callable[[gp.PrimitiveTree], Callable]):
    model = compiler(individual)
    img, y = x_y
    return int(np.sign(model(img)) == np.sign(y))
