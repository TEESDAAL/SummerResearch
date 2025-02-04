from deap import gp, base
from typing import Callable
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR
from functools import partial

# cache = {}
cache_hits = 0

def evaluate(ensemble: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray, mode: str) -> tuple[float]:
    global cache_hits
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    # eturn cache[key],

    n_splits = 5
    errors = toolbox.parallel_map(
        partial(train_then_test_model, model=ensemble, compiler=toolbox.compile, xs=xs, ys=ys),
        KFold(n_splits=n_splits).split(xs)
    )

    mean_error = sum(errors) / n_splits
    # cache[key] = mean_error

    return mean_error,


def train_then_test_model(train_test_indicies, model, compiler, xs, ys) -> float:
    predictor = compiler(model)
    train_index, test_index = train_test_indicies
    X_train, X_test = xs[train_index], xs[test_index]
    y_train, y_test = ys[train_index], ys[test_index]
    predictor.fit(X_train, y_train)

    return error(predictor.predict(X_test), y_test)


def test(
        ensemble: gp.PrimitiveTree,
        toolbox: base.Toolbox,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
    ) -> tuple[float]:
    print(f"During training there were {cache_hits} cache hits")
    predictor = toolbox.compile(ensemble)

    predictor.fit(X_train, y_train)

    return error(predictor.predict(X_test), y_test),


def error(pred, truth) -> float:
    errors = list(int(p == t) for p, t in zip(pred, truth))
    return sum(errors) / len(errors)



