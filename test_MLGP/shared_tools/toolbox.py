from deap import base, creator, gp
from deap import tools
import shared_tools.gp_restrict as gp_restrict
import numpy as np, operator
from scoop import futures
import multiprocessing
from shared_tools.fitness_function import evaluate
from functools import partial

def create_toolbox(
    data_sets: dict[str, tuple[np.ndarray, np.ndarray]], pset: gp.PrimitiveSetTyped,
    parameters, evaluation_function=evaluate) -> base.Toolbox:
    """Return a toolbox for use in GP"""


    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    if parameters.use_scoop:
        toolbox.register("close_pool", do_nothing)
        toolbox.register("parallel_map", futures.map)
    else:
        pool = multiprocessing.Pool()
        toolbox.register("close_pool", partial(close_pool, pool=pool))
        toolbox.register("parallel_map", pool.map)



    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=parameters.initial_min_depth, max_=parameters.initial_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    update_evalutation_function(toolbox, evaluation_function, data_sets)

    toolbox.register("select", tools.selDoubleTournament,
                               fitness_size=parameters.tournament_size,
                               parsimony_size=parameters.parsimony_size,
                               fitness_first=True)

    toolbox.register("selectElitism", tools.selBest)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=parameters.max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=parameters.max_depth))

    return toolbox


def update_evalutation_function(toolbox, evaluation_function, data_sets):
    x_train, y_train = data_sets["train"]
    x_validation, y_validation = data_sets["validation"]
    x_test, y_test = data_sets["test"]
    toolbox.register("evaluate", evaluation_function, toolbox=toolbox, xs=x_train, ys=y_train, mode="train")
    toolbox.register("validation", evaluation_function, toolbox=toolbox, xs=x_validation, ys=y_validation, mode="val")
    toolbox.register("test", evaluation_function, toolbox=toolbox, xs=x_test, ys=y_test, mode="test")

def close_pool(pool) -> None:
    pool.close()
    pool.join()

def do_nothing(*_) -> None:
    pass
