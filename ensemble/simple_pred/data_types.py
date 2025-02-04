from typing import Callable, final, TypeVar, Generic, Self, Protocol
from collections.abc import Sequence

import numpy as np, numpy.typing as npt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor:
    pass


X = TypeVar('X', bound=np.generic, contravariant=True)
Y = TypeVar('Y', bound=np.generic)
F = TypeVar('F', bound=np.generic)

class Predictor_(Protocol, Generic[X, Y]):
    def fit(self, x_train: npt.NDArray[X], y_train: npt.NDArray[Y]) -> Self:
        ...

    def predict(self, xs: npt.NDArray[X]) -> npt.NDArray[Y]:
        ...

@final
class Model(Generic[X, F, Y]):
    def __init__(self, feature_extractor: Callable[[X], F], model: Predictor_[F, Y]) -> None:
        assert feature_extractor is not None, "feature_extractor cannot be None"
        assert hasattr(model, "fit"), "Model must have attr fit"
        assert hasattr(model, "predict"), "Model must have attr predict"

        self.feature_extractor = feature_extractor
        self.model = make_pipeline(MinMaxScaler(), model)

    def extract_features(self, xs: npt.NDArray[X]) -> npt.NDArray[F]:
        return np.array([self.feature_extractor(x) for x in xs])

    def fit(self, x_train: npt.NDArray[X], y_train: npt.NDArray[Y]) -> Self:
        features = self.extract_features(x_train)
        _ = self.model.fit(features, y_train)
        return self

    def predict(self, xs: npt.NDArray[X]) -> npt.NDArray[Y]:
        return self.model.predict(self.extract_features(xs))



def mode[T](dist: Sequence[T]) -> T:
    return max(set(dist), key=dist.count)


@final
class Ensemble(Generic[X, Y]):
    def __init__(self, *models: Predictor_[X, Y], combining_method: Callable[[Sequence[Y]], Y]=mode):
        self.models = list(models)
        self.combine = combining_method

    def fit(self, x_train: npt.NDArray[X], y_train: npt.NDArray[Y]) -> Self:
        for model in self.models:
            _ = model.fit(x_train, y_train)

        return self

    def predict(self, xs: npt.NDArray[X]) -> npt.NDArray[Y]:
        predictions = zip(*[model.predict(xs) for model in self.models])
        return np.array([self.combine(ys) for ys in predictions])


image = np.ndarray
region = np.ndarray
prediction = tuple[float, float]
binary_function = Callable[[float, float], float]
prediction_pair = tuple[prediction, prediction]
unary_function = Callable[[float], float]
scalar = float

class Predictor:
    pass

class NumTrees:
    pass

class C:
    pass


class ImgProducer:
    pass


class KernelSize:
    pass


class Std:
    pass


class Weight:
    pass


class Order:
    pass


class Frequency:
    pass


class Orientation:
    pass

class MaxDepth:
    pass
