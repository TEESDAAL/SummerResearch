from deap import gp
import random
from functools import partial
# type: ignore
from simple_pred.function_set import (
    global_hog_small, root_con, all_lbp, all_sift, maxP, gau, gauD, lbp, hog_feature, gab, laplace,
    gaussian_Laplace1, gaussian_Laplace2, sobelxy, sobelx, sobely, medianf,
    minf, maxf, meanf, weighted_add, weighted_sub, relu, sqrt, compose, starting_point
) # type: ignore
from simple_pred.data_types import (
    Ensemble, Model, FeatureExtractor, Predictor, C, NumTrees, ImgProducer,
    KernelSize, Std, Weight, Order, Orientation, Frequency, MaxDepth
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def pset():
    pset = gp.PrimitiveSetTyped('MAIN', [], Ensemble, prefix='Image')

    for ensemble_size in [3, 5, 7]:
        pset.addPrimitive(Ensemble, [Model]*ensemble_size, Ensemble, f"vote{ensemble_size}")
        pset.addPrimitive(Ensemble, [Model]*ensemble_size, Model, f"vote{ensemble_size}")

    pset.addPrimitive(Model, [FeatureExtractor, Predictor], Model, f"model")

    predictors = [
        (gen_SVC, [C]), (gen_LR, [C]),
        (gen_RFC, [NumTrees, MaxDepth]),
        (gen_ERF, [NumTrees, MaxDepth]),
    ]

    for predictor, predictor_params in predictors:
        pset.addPrimitive(predictor, predictor_params, Predictor, predictor.__name__)
    #feature concatenation

    for n in [2, 3, 4]:
        pset.addPrimitive(partial(compose, root_con), [FeatureExtractor]*n, FeatureExtractor, name=f"Roots{n}")

    # Feature Extraction

    feature_extraction_functions = [(all_sift, 'SIFT'), (all_lbp, 'LBP_hist'), (global_hog_small, 'hog_hist')]
    for feature_extractor, name in feature_extraction_functions:
        pset.addPrimitive(partial(compose, feature_extractor), [ImgProducer], FeatureExtractor, name=name)

    # pooling
    pset.addPrimitive(partial(compose, maxP), [ImgProducer, KernelSize, KernelSize], ImgProducer, name='MaxP')

    #filtering
    pset.addPrimitive(partial(compose, gau), [ImgProducer, Std], ImgProducer, name='Gau')
    pset.addPrimitive(partial(compose, gauD), [ImgProducer, Std, Order, Order], ImgProducer, name='GauD')
    pset.addPrimitive(partial(compose, gab), [ImgProducer, Orientation, Frequency], ImgProducer, name='GaborF')

    image_processing_functions = [
        (laplace, 'Lap'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (sobelxy, 'Sobel'),
        (sobelx, 'SobelX'),(sobely, 'SobelY'), (medianf, 'Med'),
        (meanf, 'Mean'), (minf, 'Min'), (maxf, 'Max'), (lbp, 'LBP'),
        (hog_feature,'HoG'), (sqrt, 'Sqrt'), (relu, 'ReLU'),
    ]

    for img_processing_function, name in image_processing_functions:
        pset.addPrimitive(partial(compose, img_processing_function), [ImgProducer], ImgProducer, name=f'{name}')

    pset.addPrimitive(partial(compose, weighted_add), [ImgProducer, Weight, ImgProducer, Weight], ImgProducer, name='W_Add')
    pset.addPrimitive(partial(compose, weighted_sub), [ImgProducer, Weight, ImgProducer, Weight], ImgProducer, name='W_Sub')

    pset.addPrimitive(starting_point, [], ImgProducer)
    # Terminals
    pset.addEphemeralConstant('Sigma', partial(random.randint, 1, 3), Std)
    pset.addEphemeralConstant('Order', partial(random.randint, 0, 2), Order)
    pset.addEphemeralConstant('Theta', partial(random.randint, 0, 7), Orientation)
    pset.addEphemeralConstant('Frequency', partial(random.randint, 0, 5), Frequency)
    pset.addEphemeralConstant('n', gen_weight, Weight)
    pset.addEphemeralConstant('KernelSize', partial(random.randrange, 2, 5, 2), KernelSize)
    pset.addEphemeralConstant('C', gen_C, C)
    pset.addEphemeralConstant('NumTrees', partial(random.randrange, 50, 501, 10), NumTrees)
    pset.addEphemeralConstant('MaxDepth', partial(random.randrange, 10, 101, 10), MaxDepth)

    return pset


def gen_C() -> float:
    return 10**(7*round(random.random(), 3) - 2)


def gen_weight() -> float:
    return round(random.random(), 3)


def gen_SVC(c: float) -> SVC:
    return SVC(C=c)

def gen_LR(c: float) -> LogisticRegression:
    return LogisticRegression(C=c)


def gen_RFC(num_trees: int, max_depth: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth)


def gen_ERF(num_trees: int, max_depth: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=num_trees, max_depth=max_depth)

