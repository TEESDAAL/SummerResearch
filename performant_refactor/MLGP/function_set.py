import numpy as np, random, operator
from deap import gp
from MLGP.data_types import Img, X, Y, Size, Region # defined by author
from scipy import ndimage
from skimage.feature import local_binary_pattern, hog
from skimage.exposure import equalize_hist
from functools import partial

def gaussian_1(left):
    return ndimage.gaussian_filter(left,sigma=1)

#gaussian filter with sigma=1 with the second derivatives
def gaussian_11(left):
    return ndimage.gaussian_filter(left, sigma=1,order=1)

#gaussian_gradient_magnitude(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gauGM(left):
    return ndimage.gaussian_gradient_magnitude(left,sigma=1)

#gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gaussian_Laplace1(left):
    return ndimage.gaussian_laplace(left,sigma=1)

def gaussian_Laplace2(left):
    return ndimage.gaussian_laplace(left,sigma=2)

#laplace(input, output=None, mode='reflect', cval=0.0)
def laplace(left):
    return ndimage.laplace(left)

#sobel(input, axis=-1, output=None, mode='reflect', cval=0.0)
def sobel_x(left):
    return ndimage.sobel(left,axis=0)

def sobel_y(left):
    return ndimage.sobel(left,axis=1)

def lbp(image):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(np.multiply(image, 255).astype(int), 8, 1.5, method='nri_uniform')
    lbp=np.divide(lbp,59)

    return lbp

def hist_equal(image):
    return equalize_hist(image, nbins=256, mask=None)

def hog_feature(image):
    img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                         transform_sqrt=False, feature_vector=True)
    return realImage

def square_region(left,x,y,windowSize):
    width,height=left.shape
    x_end = min(width, x+windowSize)
    y_end = min(height, y+windowSize)
    slice = left[x:x_end, y:y_end]
    return slice

def rect_region(left, x, y, windowSize1,windowSize2):
    width, height = left.shape
    x_end = min(width, x + windowSize1)
    y_end = min(height, y + windowSize2)
    slice = left[x:x_end, y:y_end]
    return slice



def create_pset(image_width: int, image_height: int) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], float)

    feature_construction_layer = [np.std, np.mean, np.min, np.max]
    for func in feature_construction_layer:
        pset.addPrimitive(func, [Region], float, name=func.__name__)

    binary_operators = [operator.add, operator.sub]

    for func in binary_operators:
        pset.addPrimitive(func, [float, float], float)

    image_processing_layer = [
        (hist_equal, 'Hist_Eq'), (gaussian_1, 'Gau1'), (gaussian_11, 'Gau11'),
        (gauGM, 'GauXY'), (laplace, 'Lap'), (sobel_x, 'Sobel_X'),
        (sobel_y, 'Sobel_Y'), (gaussian_Laplace1, 'LoG1'),
        (gaussian_Laplace2, 'LoG2'), (lbp, 'LBP'), (hog_feature, 'HOG'),
    ]

    for func, name in image_processing_layer:
        pset.addPrimitive(func, [Region], Region, name=name)



    # Functions  at the region detection layer
    pset.addPrimitive(square_region, [Img, X, Y, Size], Region, name='Region_S')
    pset.addPrimitive(rect_region, [Img, X, Y, Size, Size], Region, name='Region_R')


    pset.addEphemeralConstant('X', partial(random.randint, 0, image_width - 24), X)
    pset.addEphemeralConstant('Y', partial(random.randint, 0, image_height - 24), Y)

    # Changed from 20 to 24
    pset.addEphemeralConstant('Size', partial(random.randint, 24, 60), Size)

    pset.renameArguments(ARG0='Image')

    return pset
