from typing import Callable
import numpy as np
from FlexGP.shared_tools.make_datasets import y_train, x_test, y_test
import random
random.seed(0)

prediction = tuple[float, float]
image = np.ndarray

def squared_distance(t1: prediction, t2: prediction) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(model: Callable[[image], prediction], xs: np.ndarray, ys: np.ndarray) -> float:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)
    square_errors = [
        squared_distance(model(x), y) for x, y in zip(xs, ys)
    ]

    return sum(square_errors) / len(square_errors)

print(f"All zero's error: {evaluate(lambda _: (0, 0), x_test, y_test)}")

random_results = np.array([evaluate(lambda _: (random.random()*2 - 1, random.random()*2 - 1), x_test, y_test) for _ in range(30)])
print(f"Random prediction error: {random_results.mean()} +- {random_results.std()}")
print("Best random error: ", random_results.min())
aro, val = sum(a for a, _ in y_train) / len(y_train), sum(v for _, v in y_train) / len(y_train)

print(f"Results for avg prediction {(aro, val)}: {evaluate(lambda _: (aro, val), x_test, y_test)}")

