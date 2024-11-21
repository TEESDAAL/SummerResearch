# pyright: basic, reportAttributeAccessIssue=false
import functools
from deap import gp, creator, base, tools, algorithms
import random
import math
import operator
import itertools, functools

pset = gp.PrimitiveSetTyped("MAIN", [float, float], tuple, "IN")

def combine(x: float, y: float) -> tuple[float, float]:
    return x, y

pset.addPrimitive(combine, (float, float), tuple)
pset.addPrimitive(math.sin, (float,), float)
pset.addPrimitive(math.cos, (float,), float)
pset.addPrimitive(operator.add, (float, float), float)
pset.addPrimitive(operator.sub, (float, float), float)
pset.addPrimitive(operator.mul, (float, float), float)

pset.addTerminal((0, 0), tuple)
pset.addEphemeralConstant("rand", functools.partial(random.randint, -1, 1), float)

pset.renameArguments(IN0='x')
pset.renameArguments(IN1='y')
print(pset.arguments)
def pass_through(x):
    return x

pset.addPrimitive(pass_through, [Img], Img, name='img_identity')



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("make_tree", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.make_tree)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def subtract(t1: tuple[float, float], t2: tuple[float, float]) -> tuple[float, float]:
    return t1[0] - t2[0], t1[1] - t2[1]


def error(t1: tuple[float, float], t2: tuple[float, float]) -> float:
    return sum(subtract(t1, t2))


def evaluate(individual: gp.PrimitiveTree, points: list[tuple[float, float]]) -> tuple[float]:
    # combine(add(sin(x), 1), mul(y, y)
    f = lambda x, y: (math.sin(x) + 3, y**2)

    model = toolbox.compile(individual)
    square_errors = (error(f(x, y), model(x, y))**2 for (x, y) in points)
    # RMSE
    return math.sqrt(1/len(points) * sum(square_errors)),


points = list(itertools.product([i/2 for i in range(-5, 5)], repeat=2))
print(points)
toolbox.register("evaluate", evaluate, points=points)
toolbox.register("select", tools.selDoubleTournament,
                           fitness_size=7,
                           parsimony_size=1.4,
                           fitness_first=True)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



def main() -> list[creator.Individual]:
    random.seed(10)
    pop = toolbox.population(n=500)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 200)

    return pop

if __name__ == "__main__":
    final_population = main()

    best = min(final_population, key=lambda i: i.fitness.values[0])
    print(best, best.fitness.values)



