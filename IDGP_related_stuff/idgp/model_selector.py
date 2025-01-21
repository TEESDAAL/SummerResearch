import argparse, random, numpy as np
from run_gp import main
import shared_tools.parameters as params

parser = argparse.ArgumentParser(
                    prog='Model selector',
                    description='Acts as an interface to choose the model selector')

parser.add_argument("model", help="The model to select")
parser.add_argument('-s', '--seed', default=0, help="The random seed to use, defaults to 0", type=int)
parser.add_argument('-p', '--population', default=params.population, help=f"The size of the GP population, defaults {params.population}", type=int)
parser.add_argument('-g', '--generations', default=params.generations, help=f"The number of generations to train for, defaults {params.generations}", type=int)

parser.add_argument('-c', '--crossover', default=params.cxProb, help=f"The crossover probability, defaults to {params.cxProb}", type=float)
parser.add_argument('-m', '--mutation', default=params.mutProb, help=f"The mutation probability, defaults to {params.mutProb}.", type=float)
parser.add_argument('-e', '--elitism', default=params.elitismProb, help=f"The elitism probability, defaults to {params.elitismProb}.", type=float)

parser.add_argument('-t', '--tournament-size', default=params.tournament_size, help=f"The number of individuals in each tournament, defaults to {params.tournament_size}.", type=int)
parser.add_argument('-ps', '--parsimony-size', default=params.parsimony_size, help=f"The number of individuals to participate in the selection tournament [1, 2], defaults to {params.parsimony_size}.", type=float)
parser.add_argument('-imin', '--initial-min-depth', default=params.initialMinDepth, help=f"The initial minimum depth to construct a tree, defaults to {params.initialMinDepth}")
parser.add_argument('-imax', '--initial-max-depth', default=params.initialMaxDepth, help=f"The initial minimum depth to construct a tree, defaults to {params.initialMaxDepth}")
parser.add_argument('--no-record', action='store_true', help="a keyword argument that can be passed in to not record the run.")
parser.add_argument('--no-scoop', action='store_true', help="a keyword argument that can be passed in to use the normal multithreading library instead of scoop.")


if __name__ == "__main__":
    parameters = parser.parse_args()
    random.seed(parameters.seed)
    np.random.seed(parameters.seed)
    print(parameters)
    main(parameters)

