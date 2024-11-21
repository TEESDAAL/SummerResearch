#python packages
import math
import random
import time
import numpy as np
# deap package
from deap import tools, algorithms# fitness function
# from fitnessEvaluation import evaluate, test
from make_datasets import x_test, y_test
from parameters import  population, cxProb, mutProb, generations
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

    #pop, log, hof2 =  evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generations, stats=mstats, halloffame=hof, verbose=True)
    pop, log = algorithms.eaSimple(pop, toolbox, cxProb, mutProb, generations, stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof




if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = main(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    testResults = toolbox.test(hof[0])
    testTime = time.process_time() - endTime

    print('Best individual ', hof[0])
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')
