from deap import gp
from typing import Callable
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR


def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(individual: gp.PrimitiveTree, compiler: Callable[[gp.PrimitiveTree], Callable], xs: np.ndarray, ys: np.ndarray) -> tuple[float]:
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    feature_extractor = compiler(individual)

    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)
    transformed_training_set = MinMaxScaler().fit_transform(np.array([feature_extractor(x) for x in xs]))

    total_error = 0
    for train_index, test_index in KFold(n_splits=5).split(transformed_training_set):
        X_train, X_test = transformed_training_set[train_index], transformed_training_set[test_index]
        y_train, y_test = ys[train_index], ys[test_index]
        predictor = model(random_state=0)
        predictor.fit(X_train, y_train)
        total_error += error(predictor.predict(X_test), ys)

    return total_error / 5,


def test(
        individual: gp.PrimitiveTree,
        compiler: Callable[[gp.PrimitiveTree], Callable],
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
    ):
    feature_extractor = compiler(individual)
    scaler = MinMaxScaler()
    transformed_training_set = scaler.fit_transform(np.array([feature_extractor(x) for x in X_train]))
    transformed_test_set = scaler.transform(np.array([feature_extractor(x) for x in X_test]))

    predictor = model()
    predictor.fit(transformed_training_set, y_train)
    return error(predictor.predict(transformed_test_set), y_test),


def error(pred, truth) -> float:
    errors = list(squared_distance(p, t) for p, t in zip(pred, truth))
    return sum(errors) / len(errors)


def model(**kwargs):
    # return MultiOutputRegressor(SVR(**kwargs))
    return DecisionTreeRegressor(**kwargs)

