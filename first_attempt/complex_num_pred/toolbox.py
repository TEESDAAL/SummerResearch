
from typing import Callable
from deap import base, creator, gp, tools
import gp_restrict
from function_set import pset
from parameters import initialMaxDepth, initialMinDepth
import numpy as np
from make_datasets import x_train, y_train, x_validation, y_validation, x_test, y_test
from data_types import prediction


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def squared_distance(c1: prediction, t: tuple[float, float]) -> float:
    distance = c1 - complex(*t)
    return distance.real**2 + distance.imag**2


def evaluate(individual: gp.PrimitiveTree, compiler: Callable[[gp.PrimitiveTree], Callable], x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
    """Compute the MSE  between the models answer and the true (val, aro) pair. Uses square distances avoids computing the sqrt"""
    model = compiler(individual)
    # calculate errors by MSE, error of each model given by geometric distance between values (pythag)
    square_errors = [
        squared_distance(model(x), y) for x, y in zip(x_train, y_train)
    ]

    return sum(square_errors) / len(square_errors),


toolbox.register("evaluate", evaluate, compiler=toolbox.compile, x_train=x_train, y_train=y_train)
toolbox.register("validation", evaluate, compiler=toolbox.compile, x_train=x_validation, y_train=y_validation)
toolbox.register("test", evaluate, compiler=toolbox.compile, x_train=x_test, y_train=y_test)



# toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("select", tools.selDoubleTournament,
                           fitness_size=7,
                           parsimony_size=1.4,
                           fitness_first=False)
toolbox.register("selectElitism", tools.selBest)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


