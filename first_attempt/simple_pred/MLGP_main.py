print("Program Started")
#python packages
import random
import time
import numpy as np
# deap package
import eval_gp
from deap import tools
from parameters import  population, cxProb, mutProb,  elitismProb, generations
from toolbox import toolbox
randomSeeds = 12



def main(randomSeeds):
    random.seed(randomSeeds)
    pop = toolbox.population(population)
    hof = tools.HallOfFame(1)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log, hof2 =  eval_gp.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generations, stats=mstats, halloffame=hof, verbose=True)
    # pop, log = algorithms.eaSimple(pop, toolbox, cxProb, mutProb, generations, stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof, hof2

# def evaluate(individual: gp.PrimitiveTree, x_train: np.ndarray, y_train: np.ndarray) -> tuple[float]:
#     model = toolbox.compile(individual)
#
#     square_errors = [(model(img) - float(val))**2 for img, (val, arousal) in zip(x_train, y_train)]
#     return math.sqrt(sum(square_errors) / len(square_errors)),
#
# toolbox.register("evaluate", evaluate, x_train=x_train, y_train=y_train)
print("Running GP")
if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof, hof2 = main(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    testResults = toolbox.test(hof2[0])
    testTime = time.process_time() - endTime

    print('Best individual ', hof[0])
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')
