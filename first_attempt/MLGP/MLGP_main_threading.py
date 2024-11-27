#python packages
import random, pickle
import time
import numpy as np
import eval_gp
# deap package
from deap import tools, algorithms
from display_statistics import plot
from parameters import  population, cxProb, mutProb, elitismProb, generations
from toolbox import toolbox
import multiprocessing
from scoop import futures
from random_seed import seed
import fitness_function
from make_datasets import x_train, y_train, x_test, y_test, x_validation, y_validation


def main(seed, write_to_file=False, display=False, train_for_arousal=True):
    random.seed(seed)
    np.random.seed(seed)



    evaluation_function = fitness_function.evaluate_arousal if train_for_arousal else fitness_function.evaluate_valence

    toolbox.register("evaluate", evaluation_function, compiler=toolbox.compile, x_train=x_train, y_train=y_train)
    toolbox.register("validation", evaluation_function, compiler=toolbox.compile, x_train=x_validation, y_train=y_validation)
    toolbox.register("test", evaluation_function, compiler=toolbox.compile, x_train=x_test, y_train=y_test)


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
    #pop, log = algorithms.eaSimple(pop, toolbox, cxProb, mutProb, generations, stats=mstats, halloffame=hof, verbose=True)


    if write_to_file:
        pickle.dump(log, open(f"data/{seed}{['v', 'a'][train_for_arousal]}.pkl", "wb"))

    if display:
        plot(log)


    return pop, log, hof, hof2




if __name__ == "__main__":
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    #toolbox.register("map", futures.map)


    models = []
    for with_arousal in [True, False]:
        beginTime = time.process_time()
        pop, log, hof, hof2 = main(seed(), write_to_file=True, display=False, train_for_arousal=with_arousal)
        endTime = time.process_time()
        trainTime = endTime - beginTime

        testResults = toolbox.test(hof2[0])
        testTime = time.process_time() - endTime

        print('Best individual ', hof[0])
        print('Test results  ', testResults)
        print('Train time  ', trainTime)
        print('Test time  ', testTime)
        print('End')

        models.append(toolbox.compile(hof2[0]))

    print(f"Combined fitness: {fitness_function.evaluate(lambda img: (models[0](img), models[1](img)), x_test, y_test)}")
