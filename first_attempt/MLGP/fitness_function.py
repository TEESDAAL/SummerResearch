import numpy as np
from deap import gp
from deap.base import Toolbox
import math

def evaluate(toolbox: Toolbox, individual: gp.PrimitiveTree, x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
    model = toolbox.compile(individual)

    square_errors = [(model(img) - float(val))**2 for img, (val, arousal) in zip(x_train, y_train)]
    return math.sqrt(sum(square_errors) / len(square_errors)),

def test(individual, toolbox: Toolbox):
    model = toolbox.compile(individual)
    return 3,
