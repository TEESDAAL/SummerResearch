import random, time, pickle, numpy as np
from shared_tools import toolbox
from simple_pred.make_pset import create_pset
from shared_tools.make_datasets import x_train, y_train, x_validation, y_validation, x_test, y_test
from deap import algorithms, gp, tools
import shared_tools.eval_gp as eval_gp
import simple_pred.function_set as simple_fs
from shared_tools.toolbox import create_toolbox, update_evalutation_function

def get_pset(model):
    image_width, image_height = x_train[0].shape
    pset_dict = {
        "simple_pred": create_pset,
    }

    return pset_dict[model](image_width, image_height)


def run_gp(parameters, toolbox):
    pop = toolbox.population(parameters.population)
    #toolbox.evaluate(pop)
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
    # pop, log, hof2 = eval_gp.eaSimple(
    #     pop, toolbox, parameters.crossover, parameters.mutation,
    #     parameters.elitism, parameters.generations,
    #     stats=mstats, halloffame=hof, verbose=True
    # )
    #
    # return pop, log, hof, hof2
    log = algorithms.eaSimple(
        pop, toolbox, parameters.crossover, parameters.mutation,
        parameters.generations, stats=mstats, halloffame=hof, verbose=True
    )
    print(f"{toolbox.cache_hits.value} cache hits")

    return pop, log, hof


def main(parameters, **kwargs) -> gp.PrimitiveTree:
    datasets = {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test)
    }


    pset = get_pset(parameters.model)

    toolbox = create_toolbox(datasets, pset, parameters, **kwargs)

    return record_run(parameters, toolbox, **kwargs)



def record_run(parameters, toolbox, prefix="") -> gp.PrimitiveTree:
    beginTime = time.process_time()

    pop, log, hof = run_gp(parameters, toolbox)

    endTime = time.process_time()
    trainTime = endTime - beginTime

    best_individual = hof[0]
    testResults = toolbox.test(best_individual)
    testTime = time.process_time() - endTime
    print('Best individual ', best_individual)
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')

    if not parameters.no_record:
        filepath = f"{parameters.model}/data"
        pickle.dump(log, open(f"{filepath}/{parameters.seed}-{prefix}log.pkl", 'wb'))
        pickle.dump(best_individual, open(f"{filepath}/{parameters.seed}-{prefix}best.pkl", 'wb'))
        pickle.dump(trainTime, open(f"{filepath}/{parameters.seed}-{prefix}trainTime.pkl", 'wb'))


    return hof[0]

