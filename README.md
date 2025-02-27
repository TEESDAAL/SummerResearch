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
A feature extraction approach that works by selecting and then transforming a given region, where the region and transformation(s) are all learned from the GP process. A few attempts were made with various different "wrapper models" with random forest being the best here.

#### Results
https://github.com/TEESDAAL/SummerResearch/blob/main/IDGP_related_stuff/idgp_tree/data_visualisation.ipynb

Test Error ± 1 std: 0.4381 ± 0.0049

### FlexGP
A complex feature extraction model that combines image transformation functions, filtering and pooling to reduce an image down to an array of features.
**Note:** This technique is *incredibly* slow for large images. I would recommend shrinking them down to a much smaller size (I did 32x32) so that the evaluation finishes in a reasonable amount of time. With the image size I was using before scaling it was reasonably likely for feature extraction to create >100,000 features! Causing the model to take ages to train, and reducing accuracy (see curse of dimensionality). If this is not possible I would look at removing the "FeaCon" image functions as they simply flatten and concatenate some number of images.

This model had a significant overfitting problem.

#### Results
https://github.com/TEESDAAL/SummerResearch/blob/main/FlexGP/data_visualisation.ipynb

Test Error ± 1 std: 0.4526 ± 0.0044

### COGP
A feature extraction model that aims to learn convolution and pooling operators to reduce image(s) into a feature array. Potentially it is worth exploring introducing some other feature extraction functions rather than just concatenating two or more images together.

#### Results
https://github.com/TEESDAAL/SummerResearch/blob/main/COGP/data_visualisation.ipynb

Test Error ± 1 std: 0.47337 ± 0.0067

### Ensemble Methods
Many different ensemble methods were attempted in this project.
1. EnsembleMLGP - ensemble together multiple of the MLGP models and average their predictions.
  - Low degree of success in this, presumably because of the weakness of the MLGP base learner coupled with the very small ensemble size enforced by compute time and memory. Also large runs would be run out of memory :(
2. SkLearn model overfitting. This approach aimed to combine the votes of more powerful learners to get the final prediction.
  - This model was unable to run, as the memory usage when breeding skyrocketed causing the program to crash for some reason. This is still worth investigating if the memory issue can be solved.

### Automatic Region Detection.
An attempt at using clustering for automatic region detection. I think this approach is very interesting and worth investigating further. The basic idea is that it uses a series of image transformations to increase the values of important regions, then by we convert the transformed image into a bitmap of pixels above some learned threashold value.

For this examples I invented? which I will call *sliding window clustering* (May already be a technique that exists, I'm just not aware of it). That starts with a guess for the size of the cluster and so creates a window of a given (w x h). It then slides this window across the image, keeping track of how many points fall inside that that window. Then it returns the x, y location of the window that captured the most points.

This however has no idea of centering, ideally the cluster would be placed in the center of the of the window, to achieve this I weighted the window using a 2D gaussian, so that points in the center of the window would be weighted higher. You could extend this technique to different filters/weights applied to each cell of this window to look for different cluster shapes.

Example of the threashold idea can be seen here:
https://github.com/TEESDAAL/SummerResearch/blob/main/region_detection/visualizer.ipynb


## Future Work
- Investigate the massive overfitting problem with FlexGP & COGP. Some very naive feature reduction approaches were attempted here (PCA and reducing images), and trying different models. But reducing this issue could potentially dramatically improve results.
- Try and solve the memory issues with ensemble methods.
- Further experimenting with the automatic region-detection could be interesting

### Issues with this research
- One potential issue with this work is that the validation error could "bleed over". This is because the function that adds a model to the validation hall of fame reassigns it's fitness value to it's validation error, and doesn't clone the individual. This can be fixed by either making sure to clone all the models before they're passed into the function. Or for the "evalValidation" function to clone the models passed into them.

```python
def evalValidation(offspring_for_va, toolbox, validation_hof):
    offspring_for_va = [toolbox.clone(o) for o in offspring_for_va]

    fitnesses2 = toolbox.map(toolbox.validation, offspring_for_va)
    for ind2, fit2 in zip(offspring_for_va, fitnesses2):
        ind2.fitness.values = fit2
        # Update the hall of fame with the generated individuals

    if validation_hof is not None:
        validation_hof.update(offspring_for_va)

    return validation_hof
```

- Deserializing runs proved to be very tricky only seems to be able to deserialize on the "enviroment" it was run on. In future make sure to set up a standard virtual enviroment, which might fix things?


# Notes for future runs:
- Everything that gets passed into a function that gets run across multiple processes needs to be "pickleable", this causes problems with passing the toolbox around, e.g. using the toolbox.compile function for the evaluation of a function. This can be solved by passing in the function `toolbox.compile` into the function then calling that.
- In practice I found that there was performance improvements if I didn't run the model evaluations in parallel, but ran parts of evaluation process in parallel, e.g. the image processing. To achieve this I added a `parallel_map` that used the multiprocessing `Pool().map` function (as the regular `toolbox.map` function is used by the library code).

