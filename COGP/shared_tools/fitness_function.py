import numpy as np, time
from deap import gp, base
from typing import Callable, Iterable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from functools import partial
from sklearn.decomposition import PCA

cache = {}

def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(individual: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray) -> tuple[float]:
    global cache_hits
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    key = str(individual)
    if key in cache:
        return cache[key],

    features = np.array(list(toolbox.parallel_map(
        partial(extract_features, individual=individual, compiler=toolbox.compile),
        xs
    )))

    n_splits = 5
    errors = toolbox.parallel_map(
        partial(train_then_test_model, xs=features, ys=ys),
        KFold(n_splits=n_splits).split(features)
    )

    mean_error = sum(errors) / n_splits
    cache[key] = mean_error

    return mean_error,


def train_then_test_model(train_test_indicies, xs, ys) -> float:
    train_index, test_index = train_test_indicies
    X_train, X_test = xs[train_index], xs[test_index]
    y_train, y_test = ys[train_index], ys[test_index]
    predictor = model()
    predictor.fit(X_train, y_train)

    return error(predictor.predict(X_test), y_test)


def model():
    return make_pipeline(
        MinMaxScaler(),
        #PCA(n_components=10),
        #LinearRegression()
        RandomForestRegressor(random_state=0)
        #MultiOutputRegressor(LinearSVR(random_state=0))
        #DecisionTreeRegressor(random_state=0, max_depth=7)
    )

def validate(
        individual: gp.PrimitiveTree,
        toolbox: base.Toolbox,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
) -> tuple[float]:
    val_error, _ = test(individual, toolbox, X_train, y_train, X_test=X_val, y_test=y_val)
    return val_error,

def test(
        individual: gp.PrimitiveTree,
        toolbox: base.Toolbox,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
    ) -> tuple[float, float]:
    """Train the final model, returns a tuple of the (test_error, train_error)"""
    train_features = np.array(list(toolbox.map(
        partial(extract_features, individual=individual, compiler=toolbox.compile),
        X_train
    )))
    test_features = np.array(list(toolbox.map(
        partial(extract_features, individual=individual, compiler=toolbox.compile),
        X_test
    )))
    predictor = model()
    predictor.fit(train_features, y_train)
    return error(predictor.predict(test_features), y_test), error(predictor.predict(train_features), y_train)


def error(pred: Iterable[tuple[float, float]], truth: Iterable[tuple[float, float]]) -> float:
    errors = list(squared_distance(p, t) for p, t in zip(pred, truth))
    return sum(errors) / len(errors)

def extract_features(image, individual, compiler) -> list[float]:
    return compiler(individual)(image)


