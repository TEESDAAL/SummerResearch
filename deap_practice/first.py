from deap import base, creator, tools, algorithms
from functools import partial
import operator
import numpy
from deap import gp
import math
import random
from deap.gp import PrimitiveSet, PrimitiveTree, genFull
from networkx.drawing.nx_agraph import graphviz_layout


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMin)

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

pset.addEphemeralConstant("rand101", partial(random.choice, (-1, 0, 1)))

pset.renameArguments(ARG0='x')



toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points) -> tuple[float]:
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the given function
    return math.fsum((func(x) - goal_function(x))**2 for x in points) / len(points),


def goal_function(x: float) -> float:
    return math.sin(x)


toolbox.register(
    "evaluate",
    evalSymbReg,
    points=[x/50. for x in range(-100,100)],
)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    population, _, _ = main()
    print(min(population, key=lambda i: i.fitness.values[0]))
