import numpy as np, random
from deap import gp
from functools import partial
from simple_pred.data_types import (
    Img, FilteredImg, Regions, LowerBound, NumClusters, Padding,
)
from simple_pred.function_set import (
    hist_equal, gaussian_1, gaussian_11, gauGM, laplace, sobel_x, sobel_y,
    gaussian_Laplace1, gaussian_Laplace2, threashold,
    centroid_clustering_to_region
)
from sklearn.cluster import KMeans

def create_pset(image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Regions)

    image_combination = [np.add, np.subtract]
    for func in image_combination:
        pset.addPrimitive(func, [Img, Img], Img, name=func.__name__)


    image_processing_layer = [
        (hist_equal, 'Hist_Eq'), (gaussian_1, 'Gau1'), (gaussian_11, 'Gau11'),
        (gauGM, 'GauXY'), (laplace, 'Lap'), (sobel_x, 'Sobel_X'),
        (sobel_y, 'Sobel_Y'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (abs, 'abs')
    ]

    for func, name in image_processing_layer:
        pset.addPrimitive(func, [Img], Img, name=name)

    pset.addPrimitive(threashold, [Img, LowerBound], FilteredImg)
    pset.addEphemeralConstant('lower_bound', random.random, LowerBound)

    # image: image, number_clusters: int, x_pad: int, y_pad: int, clustering_method
    pset.addPrimitive(
        partial(centroid_clustering_to_region, clustering_method=KMeans),
        [FilteredImg, NumClusters, Padding, Padding], Regions,
        name="KMeans"
    )


    #pset.addEphemeralConstant('l_padding', partial(random.randint, image_width // 10, image_width // 5), Padding)
    pset.addEphemeralConstant('l_padding', partial(random.randint, 5, 12), Padding)

    #pset.addEphemeralConstant('clusters', partial(random.randint, 1, 7), NumClusters)
    pset.addTerminal(3, NumClusters)
    # Changed from 20 to 24
    pset.renameArguments(ARG0='Image')

    return pset



