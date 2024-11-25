from make_datasets import x_train

population = 100
generations = 5
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 6
image_width, image_height = x_train[0].shape
