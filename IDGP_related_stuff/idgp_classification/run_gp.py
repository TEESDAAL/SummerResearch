import argparse, time, pickle, numpy as np
from shared_tools.make_datasets import x_train, y_train, x_test, y_test, x_validation, y_validation
from deap import algorithms, gp, tools
import shared_tools.eval_gp as eval_gp
import simple_pred.function_set as simple_fs
from shared_tools.toolbox import create_toolbox
from dataclasses import dataclass


def get_pset(model):
    image_width, image_height = x_train[0].shape
    pset_dict = {
        "simple_pred": simple_fs.create_pset,
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
    pop, log, hof2 =  eval_gp.eaSimple(
        pop, toolbox, parameters.crossover, parameters.mutation,
        parameters.elitism, parameters.generations,
        stats=mstats, halloffame=hof, verbose=True
    )

    return pop, log, hof, hof2


def main(parameters, **kwargs) -> gp.PrimitiveTree:
    datasets = {
        "train": (x_train, y_train),
        "val": (x_validation, y_validation),
        "test": (x_test, y_test)
    }

    pset = get_pset(parameters.model)

    toolbox = create_toolbox(datasets, pset, parameters, **kwargs)

    return record_run(parameters, toolbox, **kwargs)



def record_run(parameters, toolbox, prefix="") -> gp.PrimitiveTree:
    begin_time = time.process_time()

    pop, log, hof, hof2 = run_gp(parameters, toolbox)

    end_time = time.process_time()
    train_time = end_time - begin_time

    best_individual = hof[0]
    testResults = toolbox.test(best_individual)
    test_time = time.process_time() - end_time
    print('Best individual ', best_individual)
    print('Test results  ', testResults)
    print('Train time  ', train_time)
    print('Test time  ', test_time)
    print('End')

    if not parameters.use_scoop:
        toolbox.close_pool()

    if parameters.no_record:
        return hof[0]

    filepath = "simple_pred/data"
    run_info = RunInfo(
        "IDGP_FULL_AFFECT", parameters, best_individual, log,
        hof, hof2, train_time, test_time, pop
    )

    with open(f"{filepath}/{parameters.seed}-run_info.pkl", 'wb') as log_file:
        pickle.dump(run_info, log_file)


    return hof[0]


@dataclass
class RunInfo:
    model: str
    parameters: argparse.ArgumentParser
    best_individual: gp.PrimitiveTree
    log: tools.Logbook
    hall_of_fame: tools.HallOfFame
    val_hall_of_fame: tools.HallOfFame
    train_time: float
    test_time: float
    final_population: list[gp.PrimitiveTree]
