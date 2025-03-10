import matplotlib, os, pickle
from shared_tools.make_datasets import x_train, y_train
from shared_tools.fitness_function import evaluate
from complex_num_pred.fitness_function import evaluate as complex_eval
from functools import partial
from run_gp import get_pset
from deap import gp
import multiprocessing

MODELS = ('complex_num_pred', 'complex_pred', 'simple_pred')
from matplotlib import pyplot as plt
def main(pool):
    plt.boxplot([obtain_all_results(model, pool) for model in MODELS])
    plt.show()


def models(dir_path: str) -> list[gp.PrimitiveTree]:
    path = f"{dir_path}/data"
    return [pickle.load(open(f"{path}/{file}", 'rb')) for file in os.listdir(path) if 'best' in file]



def obtain_all_results(model: str, pool) -> list[float]:
    evaluation_function = complex_eval if model == "complex_num_pred" else evaluate
    compiler = partial(gp.compile, pset=get_pset(model))
    return [fitness[0] for fitness in pool.map(
            partial(evaluation_function, compiler=compiler, x_train=x_train, y_train=y_train),
            models(model)
        )]

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    main(pool)
