# Summer Research
This is a repository for all my summer research related things.

## Running the models
All of the models *should* follow the same process to run them. Once you've `cd`ed into a specific folder it can be run in the following ways:
```sh
python main.py
```
to run with the default testing parameters (usually a population size and number of generations of 2, these can be customized with the `-p` and `-g` commands respectively).

To see the full set of customization options run 
```sh
python main.py --help
```

Also you can run without multithreading by telling it to use scoop without running it using scoop
```sh
python main.py --use-scoop
```

## Approaches
### MLGP
One of the simplest approaches, just getting the std of the pixels of a transformed image. Because of this simplicity the model results weren't very good, especially with the limitations of std being only positive caused this model to perform quite poorly with respect to this task. A very brief attempt to address this problem was attempted in [test_MLGP](https://github.com/TEESDAAL/SummerResearch/tree/main/test_MLGP) (by simply sifting the output of the std function by the average std of the images, then scaling the output so it was in the range (1, -1). This made a very minimal difference to overall performance. 

#### Results
https://github.com/TEESDAAL/SummerResearch/blob/main/MLGP/data_visualisation.ipynb
Test Error ± 1 std: 0.5209 ± 0.0020

### IDGP
