# Summer Research
This is a repository for all my summer research related things.

## Running the models
Running the models is now done through `model_selector.py` in the `refactor` directory. You can run a specific model in the following ways:
1. ```python -m scoop model_seletor.py MODEL [parameters]``` - Run the model using scoop for parallelism.
2. ```python model_seletor.py MODEL [parameters] --no-scoop``` - Run the model using the regular multithreading library for parrallelism.
3. ```python model_seletor.py MODEL [parameters]``` -  Run the model without any parrallelism


For more details you can run `python model_selector --help`

## Example
Eg:

`python model_selector.py complex_pred -g 2 -p 10` - Run the complex_pred model, trained for 2 generations with a population size of 10.
## Explanation of the models
### "Seperate" MLGP Model
Run via `python -m scoop model_selector.py MLGP`
The simpliest model. Goes through the learning process twice. Once to produce a tree that predicts the valence, and once to produce the values for the arousal.
These trees are then simply combined.

### "Simple" MLGP Model (`simple_pred` directory):
Run via `python -m scoop model_selector.py MLGP`
A reasonably simple model. Unline the previous model, the whole tree is trained together in one run. And the error of each model is given by the euclidean distance between the prediction and the true value.

### Complex Number Model (`complex_num_pred` directory):
Run via `python -m scoop complex_number_pred.py MLGP`
Very much like the previous model but the outputs are in the form of complex numbers (a+bi) instead of a tuple. The program also has a small "regression" layer where the output can be scaled or rotated (by multiplying the output by a complex number).

### Complex Regression MLGP mode (`complex_pred` directory):
Run via `python -m scoop complex_pred.py MLGP`
Adds a regression layer on top after the prediction. Allows for the intermediate values predicted by the previous layers to be recombined in a bunch of different ways.

Assuming the lower layers produce some intermediate value (x, y). The regression layer allows for these to be combined in a more complex fashion. Eg. (x^2+y, sin(x)*y)

