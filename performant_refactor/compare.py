import matplotlib, os, pickle
from shared_tools.make_datasets import x_train, y_train
from shared_tools.fitness_function import evaluate
from functools import partial
from run_gp import get_pset
from deap import gp
import multiprocessing

MODELS = ('complex_num_pred', 'complex_pred', 'simple_pred')
from matplotlib import pyplot as plt
def main(pool):
    plt.boxplot([obtain_all_results(model, pool) for model in MODELS])


def models(dir_path: str) -> list[gp.PrimitiveTree]:
    return [pickle.load(open(file, 'rb')) for file in os.listdir(f"{dir_path}/data") if 'best' in file]



def obtain_all_results(model: str, pool) -> list[float]:
    compiler = partial(gp.compile, pset=get_pset(model))
    return [fitness[0] for fitness in pool.map(
            partial(evaluate, compiler=compiler, x_train=x_train, y_train=y_train),
            models(model)
        )]

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    main(pool)
