from deap import base, creator, gp
from deap import tools
from scipy.ndimage import value_indices
import shared_tools.gp_restrict as gp_restrict
import numpy as np, operator, multiprocessing
from scoop import futures
from shared_tools.fitness_function import evaluate

def create_toolbox(
    data_sets: dict[str, tuple[np.ndarray, np.ndarray]], pset: gp.PrimitiveSetTyped,
    parameters) -> base.Toolbox:
    """Return a toolbox for use in GP"""


    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    if not parameters.use_scoop:
        pool = multiprocessing.Pool()
        toolbox.register("parallel_map", pool.map)
    else:
        toolbox.register("parallel_map", futures.map)



    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=parameters.initial_min_depth, max_=parameters.initial_min_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("cache_hits", Box(0))

    update_evalutation_function(toolbox, data_sets)

    toolbox.register("select", tools.selDoubleTournament,
                               fitness_size=parameters.tournament_size,
                               parsimony_size=parameters.parsimony_size,
                               fitness_first=True)

    toolbox.register("selectElitism", tools.selBest)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # TODO CHECK THIS DOESN'T BREAK ANYTHING
    toolbox.decorate("expr", gp.staticLimit(key=operator.attrgetter("height"), max_value=30))
    toolbox.decorate("expr_mut", gp.staticLimit(key=operator.attrgetter("height"), max_value=80))

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=80))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=80))

    return toolbox


def update_evalutation_function(toolbox, data_sets):
    x_train, y_train = data_sets["train"]
    x_validation, y_validation = data_sets["validation"]
    x_test, y_test = data_sets["test"]
    toolbox.register("evaluate", evaluate, toolbox=toolbox, xs=x_train, ys=y_train, mode="train")
    toolbox.register("validation", evaluate, toolbox=toolbox, xs=x_validation, ys=y_validation, mode="val")
    toolbox.register("test", evaluate, toolbox=toolbox, xs=x_test, ys=y_test, mode="test")


class Box:
    def __init__(self, value):
        self.value = value
    def __call__(self):
        # This is so it can go into the toolbox, don't call this function
        return self.value
