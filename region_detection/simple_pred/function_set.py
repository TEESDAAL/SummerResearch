from collections.abc import Callable
from functools import partial
import numpy as np
from deap import gp
from shared_tools.fitness_function import Landmarks
from simple_pred.data_types import (
    image, filtered_image, region, Size, Regions, Img, Padding, NumClusters,
    SmallPadding, Scorer, LowerBound, FilteredImg, Weight, WindowSize
)

from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.feature import local_binary_pattern, hog
from skimage.exposure import equalize_hist
import random
from itertools import product
from sklearn.cluster import DBSCAN, KMeans, Birch, AgglomerativeClustering


def gaussian_1(img_region: image) -> image:
    return ndimage.gaussian_filter(img_region, sigma=1)


#gaussian filter with sigma=1 with the second derivatives
def gaussian_11(img_region: image) -> image:
    return ndimage.gaussian_filter(img_region, sigma=1, order=1)


#gaussian_gradient_magnitude(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gauGM(img_region: image) -> image:
    return ndimage.gaussian_gradient_magnitude(img_region, sigma=1)


#gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gaussian_Laplace1(img_region: image) -> image:
    return ndimage.gaussian_laplace(img_region, sigma=1)


def gaussian_Laplace2(img_region: image) -> image:
    return ndimage.gaussian_laplace(img_region, sigma=2)


#laplace(input, output=None, mode='reflect', cval=0.0)
def laplace(img_region: image) -> image:
    return ndimage.laplace(img_region)


#sobel(input, axis=-1, output=None, mode='reflect', cval=0.0)
def sobel_x(img_region: image) -> image:
    return ndimage.sobel(img_region, axis=0)


def sobel_y(img_region: image) -> image:
    return ndimage.sobel(img_region, axis=1)


def lbp(img_region: image) -> image:
    # 'uniform','default','ror','var'
    return np.divide(
        local_binary_pattern(np.multiply(img_region, 255).astype(int), 8, 1.5, method='nri_uniform'),
        59
    )

def lbp2(img_region: image) -> image:
    # 'uniform','default','ror','var'
    return np.divide(
        local_binary_pattern(img_region, 8, 1.5, method='nri_uniform'),
        59
    )


def hist_equal(img_region: image) -> image:
    return equalize_hist(img_region, nbins=256, mask=None)


def hog_feature(img_region: image) -> image:
    img, realImage = hog(img_region, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                         transform_sqrt=False, feature_vector=True)
    return realImage


def rescale(img_region: image) -> image:
    if img_region.max() - img_region.min() == 0:
        return img_region * 0

    return (img_region - img_region.min()) / (img_region.max() - img_region.min())


def square_region(img_region: image, x: int, y: int, window_size: int) -> image:
    width, height = img_region.shape
    x_end = min(width, x + window_size)
    y_end = min(height, y + window_size)
    return img_region[x:x_end, y:y_end]


def rect_region(img_region: image | filtered_image, x: int, y: int, width: int, height: int) -> image | filtered_image:
    w, h = img_region.shape
    x_end = min(w, x + width)
    y_end = min(h, y + height)
    return img_region[y:y_end, x:x_end]


def centroid_clustering_to_region(image: filtered_image, number_clusters: int, x_pad: int, y_pad: int, clustering_method) -> list[region]:
    if number_clusters == 0:
        return []

    clusterer = clustering_method(n_clusters=number_clusters)
    img_w, img_h = image.shape
    points = np.array([p for p in product(range(img_w), range(img_h)) if image[p] != 0])
    if len(points) < number_clusters:
        # Make the cluster the size of the whole image if it there are no valid points
        return [(0, 0, 0, 0)]*number_clusters

    clusterer.fit(points)

    regions = []
    centroids = clusterer.cluster_centers_ if clustering_method == KMeans else clusterer.subcluster_centers_

    for cx, cy in centroids:
        top_x, top_y = max(0, cx - x_pad), max(0, cy - y_pad)
        max_width, max_height = img_w - cx, img_h - cy

        width, height = min(2*x_pad, max_width), min(2*y_pad, max_height)
        regions.append((int(top_x), int(top_y), int(width), int(height)))

    return list(sorted(regions, key=lambda region: region[0] + region[2]/2))


def density_clustering_to_region(image: filtered_image, number_clusters: int, x_pad: int, y_pad: int, clustering_method) -> list[region]:
    if number_clusters == 0:
        return []

    img_w, img_h = image.shape
    regions: list[region] = [(0, 0, 0, 0)]*number_clusters

    clusterer = clustering_method(n_clusters=number_clusters)
    points = np.array([p for p in product(range(img_w), range(img_h)) if image[p]])
    if len(points) < number_clusters:
        # Make the cluster the size of the whole image if it there are no valid points
        return regions

    labeled_points = zip(clusterer.fit_predict(points), points)
    clusters = {label: [] for label in clusterer.labels_}

    for label, point in labeled_points:
        if label == -1:
            continue
        clusters[label].append(point)


    for i, points in enumerate(clusters.values()):
        xs, ys = np.array([x for x, _ in points]), np.array([y for _, y in points])
        top_x, top_y = xs.min(), ys.min()
        width, height = xs.max() - top_x, ys.max() - top_y

        regions[i] = (
            min(0, top_x - x_pad), min(0, top_y - y_pad),
            max(img_w - top_x, width + x_pad), max(img_h - top_y, height + y_pad)
        )


    return list(sorted(regions, key=lambda region: region[0] + region[2]/2))



def scorer(rect_area: float, hull_area: float, linearness: float, cov_weight: float, points: Landmarks):
    xs, ys = np.array([x for x, _ in points]), np.array([y for _, y in points])
    def area():
        width, height = xs.max() - xs.min(), ys.max() - ys.min()
        return width * height

    return sum(w*f() for w, f in [
        (rect_area, area), (hull_area, lambda: ConvexHull(points).volume),
        (linearness, lambda: abs(np.cov(xs, ys)[0][1])), (cov_weight, lambda: np.cov(xs, ys)[0][1])
    ] if w != 0)


def moving_window_cluster(image: filtered_image, *dimensions: tuple[int, int]) -> list[region]:
    mutable_image = image.copy()

    return [densist_region(mutable_image, width, height) for width, height in dimensions]

def densist_region(image: filtered_image, window_width: int, window_height: int) -> region:
    image_width, image_height = image.shape
    best_x, best_y, highest_density = 0, 0, 0

    gaussian_for_centering = make_filter(window_width, window_height)
    for x in range(image_width - window_width):
        for y in range(image_height - window_height):
            captured_pixels = (rect_region(image, x, y, window_width, window_height) * gaussian_for_centering).sum()
            if captured_pixels > highest_density:
                highest_density = captured_pixels
                best_x, best_y = x, y
    # make it so the best region is "used up" so it won't be chosen again
    rect_region(image, best_x, best_y, window_width, window_height)[:] = 0
    return (best_x, best_y, window_width, window_height)


def make_filter(width: int, height: int) -> np.ndarray:
    center_x, center_y = width // 2, height // 2
    x_scale, y_scale = 1 / width, 1 / height
    return np.array([
        [np.exp(-x_scale*(x - center_x)**2 - y_scale*(y - center_y)**2) for x in range(width)]
        for y in range(height)
    ])


def threashold(img: image, lower_bound: float) -> filtered_image:
    return img > lower_bound

def fancy_threashold(img: image, threashold: float) -> image:
    flat_array = np.sort(img.flatten())
    index = int(threashold * (len(flat_array)-1))
    return img > (flat_array[index])

def create_pset(image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Regions)

    image_combination = [np.add, np.subtract]
    for func in image_combination:
        pset.addPrimitive(func, [Img, Img], Img, name=func.__name__)


    image_processing_layer = [
        (hist_equal, 'Hist_Eq'), (gaussian_1, 'Gau1'), (gaussian_11, 'Gau11'),
        (gauGM, 'GauXY'), (laplace, 'Lap'), (sobel_x, 'Sobel_X'),
        (sobel_y, 'Sobel_Y'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (abs, 'abs'), (custom_edge_detection, 'custom_edge')
    ]

    for func, name in image_processing_layer:
        pset.addPrimitive(func, [Img], Img, name=name)

    pset.addPrimitive(threashold, [Img, LowerBound], FilteredImg)
    pset.addPrimitive(fancy_threashold, [Img, LowerBound], FilteredImg)

    centroid_clustering = [KMeans, Birch]
    # image: image, number_clusters: int, x_pad: int, y_pad: int, clustering_method
    # for clustering_method in centroid_clustering:
    #     pset.addPrimitive(
    #         partial(centroid_clustering_to_region, clustering_method=clustering_method),
    #         [FilteredImg, NumClusters, Padding, Padding], Regions,
    #         name=f'{clustering_method.__name__}'
    #     )
    #
    # density_clustering = [AgglomerativeClustering]
    # for clustering_method in density_clustering:
    #     pset.addPrimitive(
    #         partial(density_clustering_to_region, clustering_method=clustering_method),
    #         [FilteredImg, NumClusters, SmallPadding, SmallPadding], Regions,
    #         name=f'{clustering_method.__name__}'
    #     )
    #
    pset.addPrimitive(
        moving_window_cluster, [FilteredImg, WindowSize, WindowSize, WindowSize], Regions
    )


    pset.addEphemeralConstant('lower_bound', gen_lower_bound, LowerBound)
    pset.addEphemeralConstant('window_size', partial(gen_window_size, image_width, image_height), WindowSize)
    #pset.addPrimitive(gen_scorer, [Weight]*4, Scorer, name="scorer")

    pset.addEphemeralConstant('s_padding', partial(random.randint, -5, 5), SmallPadding)
    pset.addEphemeralConstant('l_padding', partial(random.randint, image_width // 20, image_width // 5), Padding)
    #pset.addEphemeralConstant('l_padding', partial(random.randint, 5, 12), Padding)

    #pset.addEphemeralConstant('clusters', partial(random.randint, 1, 7), NumClusters)
    pset.addTerminal(3, NumClusters)
    # Changed from 20 to 24
    pset.addEphemeralConstant('size', partial(random.randint, 24, 60), Size)
    pset.addEphemeralConstant('weight', gen_weight, Weight)
    pset.renameArguments(ARG0='Image')

    return pset


def gen_lower_bound() -> float:
    return round(random.random(), 2)

def custom_edge_detection(image: image) -> image:
    return rescale(abs(sobel_x(image)) - abs(sobel_y(image)))

def gen_weight() -> float:
    return float(random.choice(np.arange(-2, 2.1, 0.5)))

def gen_scorer(*args) -> Callable:
    return partial(scorer, *args)

def gen_window_size(image_width, image_height) -> tuple[int, int]:
    return random.randint(image_width // 10, image_width // 2), random.randint(image_height // 10, image_height // 2)


