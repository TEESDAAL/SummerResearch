from deap import base, creator, gp
from deap import tools
import shared_tools.gp_restrict as gp_restrict
import numpy as np
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
        toolbox.register("parallel_map", futures.map)
    else:
        pool = multiprocessing.Pool()
        toolbox.register("close_pool", partial(close_pool, pool=pool))
        toolbox.register("parallel_map", pool.map)



    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=parameters.initial_min_depth, max_=parameters.initial_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", compile, pset=pset)

    update_evalutation_function(toolbox, evaluation_function, data_sets)

    toolbox.register("select", tools.selDoubleTournament,
                               fitness_size=parameters.tournament_size,
                               parsimony_size=parameters.parsimony_size,
                               fitness_first=True)

    toolbox.register("selectElitism", tools.selBest)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

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


def compile(expr, pset: gp.PrimitiveSetTyped):
    """
    Changed to work with trees that take no arguments
    Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)

    args = ",".join(arg for arg in pset.arguments) if len(pset.arguments) > 0 else ""
    code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)