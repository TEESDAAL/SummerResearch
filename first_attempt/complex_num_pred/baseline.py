from typing import Callable
import numpy as np
from data_types import image, prediction
from make_datasets import x_train, y_train, x_test, y_test
import random
from random_seed import seed
from itertools import product
random.seed(seed())

def squared_distance(c1: prediction, t: tuple[float, float]) -> float:
    c2 = complex(*t)
    distance = c1 - c2
    return distance.real**2 + distance.imag**2


def evaluate(model: Callable[[image], prediction], xs: np.ndarray, ys: np.ndarray) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)
    square_errors = [
        squared_distance(model(x), y) for x, y in zip(xs, ys)
    ]

    return sum(square_errors) / len(square_errors),

print(f"All zero's error: {evaluate(lambda _: 0 + 0j, x_test, y_test)}")
print(f"Random prediction error: {evaluate(lambda _: complex(random.random()*2 - 1, random.random()*2 - 1), x_test, y_test)}")

models = [(lambda _: complex(a, v), (a, v)) for a, v in product(map(float, np.arange(-1, 1.1, 0.1)), repeat=2)]

avg_val = sum(v for v, _ in y_train) / len(y_train)
avg_aro = sum(a for _, a in y_train) / len(y_train)


print(f"Results for best grid searched values  {(avg_val, avg_aro)}: {evaluate(lambda _: complex(avg_val, avg_aro), x_test, y_test)}")
