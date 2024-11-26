from typing import Callable
import numpy as np
from data_types import image, prediction
from make_datasets import x_train, y_train, x_test, y_test
import random
from itertools import product
random.seed(0)
def squared_distance(t1: prediction, t2: prediction) -> float:
    return (float(t1[0]) - float(t2[0]))**2 + (float(t1[1]) - float(t2[1]))**2

def evaluate(model: Callable[[image], prediction], xs: np.ndarray, ys: np.ndarray) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)
    square_errors = [
        squared_distance(model(x), y) for x, y in zip(xs, ys)
    ]

    return sum(square_errors) / len(square_errors),

print(f"All zero's error: {evaluate(lambda _: (0, 0), x_test, y_test)}")
print(f"Random prediction error: {evaluate(lambda _: (random.random()*2 - 1, random.random()*2 - 1), x_test, y_test)}")
models = [(lambda _: (a, v), (a, v)) for a, v in product(map(float, np.arange(-1, 1.1, 0.1)), repeat=2)]
best_hard_coded_values = min(models, key=lambda m: evaluate(m[0], x_train, x_test))

print(f"results for {best_hard_coded_values[1]}: {evaluate(best_hard_coded_values[0], x_test, y_test)}")
