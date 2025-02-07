import time, pickle, numpy as np, argparse
from shared_tools.make_datasets import x_train, y_train, x_validation, y_validation, x_test, y_test
from deap import gp, tools
from dataclasses import dataclass
import shared_tools.eval_gp as eval_gp
from shared_tools.toolbox import create_toolbox
from simple_pred.pset import create_pset
from typing import Callable, Iterable
from itertools import product

def run_gp(parameters, toolbox):
    print("Creating population")
    pop = toolbox.population(parameters.population)
    print("Finished creating population")

    hof = tools.HallOfFame(1)

    stats = tools.Statistics()
    metrics = [np.min, np.max, np.mean, np.std]
    values_to_track = [(lambda ind: ind.fitness.values, "fit"), (len, "size")]

    for (key, name), metric in product(values_to_track, metrics):
        stats.register(f"{name}_{metric.__name__}", add_stat_tracker(metric, key))

    stats.register(
        "val_min",
        lambda population: float(toolbox.validation(min(population, key=lambda p: p.fitness.values[0]))[0])
    )
    stats.register(
        "val_max",
        lambda population: float(toolbox.validation(max(population, key=lambda p: p.fitness.values[0]))[0])
    )

    pop, log, val_hof =  eval_gp.eaSimple(
        pop, toolbox, parameters.crossover, parameters.mutation,
        parameters.elitism, parameters.generations,
        stats=stats, hall_of_fame=hof, verbose=True
    )

    return pop, log, hof, val_hof

def add_stat_tracker[T](metric: Callable[[list[T]], float], mapper: Callable[[gp.PrimitiveTree], T]) -> Callable[[list[gp.PrimitiveTree]], float]:
    return lambda population: round(metric(list(map(mapper, population))), 5)

def main(parameters, **kwargs) -> gp.PrimitiveTree:
    datasets = {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test)
    }


    pset = create_pset(*x_train[0].shape)

    toolbox = create_toolbox(datasets, pset, parameters, **kwargs)

    return record_run(parameters, toolbox, **kwargs)



def record_run(parameters, toolbox) -> gp.PrimitiveTree:
    begin_time = time.process_time()

    pop, log, hof, val_hof = run_gp(parameters, toolbox)


    end_time = time.process_time()
    train_time = end_time - begin_time

    best_individual = val_hof[0]
    testResults = toolbox.test(best_individual)
    test_time = time.process_time() - end_time
    print('Best individual ', best_individual)
    print('Test results  ', testResults)
    print('Train time  ', train_time)
    print('Test time  ', test_time)
    print('End')
    toolbox.close_pool()

    if parameters.no_record:
        return hof[0]

    filepath = "simple_pred/data"
    run_info = RunInfo(
        "MLGP", parameters, best_individual, log,
        hof, val_hof, train_time, test_time, pop
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

