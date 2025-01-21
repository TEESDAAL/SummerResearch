from deap import gp, base
import numpy as np, numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from functools import partial
from simple_pred.function_set import rect_region

cache = {}
cache_hits = 0


def squared_distance(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2

def evaluate(individual: gp.PrimitiveTree, toolbox: base.Toolbox, xs: np.ndarray, ys: np.ndarray, mode: str) -> tuple[float]:
    global cache_hits
    """Compute the MSE of the distances between the models answer and the true (val, aro) pair avoids computing the sqrt"""
    key = str(individual) + mode
    if key in cache:
        cache_hits += 1
        return cache[key],
    features = list(toolbox.parallel_map(
        partial(extract_features, individual=individual, compiler=toolbox.compile),
        xs
    ))
    features = np.array(features)

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


def extract_features(image, individual, compiler) -> npt.NDArray[np.floating]:
    regions = compiler(individual)(image)
    return np.concatenate([rect_region(image, *region).flatten() for region in regions])


def model():
    return make_pipeline(
        MinMaxScaler(),
        SVC(random_state=0)
    )


def test(
        individual: gp.PrimitiveTree,
        toolbox: base.Toolbox,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
    ):
    global cache_hits

    print(f"There were {cache_hits} cache hits")
    train_features = np.array(list(toolbox.parallel_map(
        partial(extract_features, individual=individual, compiler=toolbox.compile),
        X_train
    )))
    test_features = np.array(list(toolbox.parallel_map(
        partial(extract_features, individual=individual, compiler=toolbox.compile),
        X_test
    )))
    predictor = model()
    predictor.fit(train_features, y_train)

    return error(predictor.predict(test_features), y_test)


def error(pred, truth) -> float:
    errors = list(int(p == t) for p, t in zip(pred, truth))
    return sum(errors) / len(errors)



