import numpy as np
from simple_pred.data_types import image, region
from scipy import ndimage
from skimage.feature import local_binary_pattern, hog
from skimage.exposure import equalize_hist
from itertools import product
from sklearn.cluster import KMeans
import numpy.typing as npt

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
    _, realImage = hog(img_region, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                         transform_sqrt=False, feature_vector=True)
    return realImage


def square_region(img: image, x: int, y: int, window_size: int) -> image:
    return rect_region(img, x, y, window_size, window_size)


def rect_region(img: image, x: int, y: int, width: int, height: int) -> image:
    w, h = img.shape
    x_end = min(w, x + width)
    y_end = min(h, y + height)
    return img[y:y_end, x:x_end]

def get_or_default(array: np.ndarray, x, y, default):
    try:
        return array[y][x]
    except IndexError:
        return default

def centroid_clustering_to_region(image: image, number_clusters: int, x_pad: int, y_pad: int, clustering_method) -> npt.NDArray[region]:
    if number_clusters == 0:
        return np.array([])

    clusterer = clustering_method(n_clusters=number_clusters)
    img_w, img_h = image.shape
    points = np.array([p for p in product(range(img_w), range(img_h)) if image[p] != 0])
    if len(points) < number_clusters:
        # Make the cluster the size of the whole image if it there are no valid points
        return np.array([(0, 0, 2*x_pad, 2*y_pad)]*number_clusters)

    clusterer.fit(points)

    regions = []
    centroids = clusterer.cluster_centers_ if clustering_method == KMeans else clusterer.subcluster_centers_

    for cx, cy in centroids:
        # top_x, top_y = max(0, cx - x_pad), max(0, cy - y_pad)
        # width, height = min(img_w, cx + 2*x_pad), min(img_h, cx + 2*y_pad)
        # regions.append((int(top_x), int(top_y), int(width), int(height)))
        top_x, top_y = cx - x_pad, cy - y_pad
        width, height = 2*x_pad, 2*y_pad
        regions.append((int(top_x), int(top_y), int(width), int(height)))
    assert len({region[2]*region[3] for region in regions}) == 1, regions
    return np.array(list(sorted(regions, key=lambda region: region[0] + region[2]/2)))

def threashold(img: image, lower_bound) -> image:
    return np.vectorize(lambda b: 1.0 if b > lower_bound else 0.0)(img)



