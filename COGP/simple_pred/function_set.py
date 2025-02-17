import simple_pred.sift_features as sift_features
import numpy as np
from scipy import ndimage
from skimage.filters import sobel
from skimage.filters import gabor
from skimage.filters import gaussian
import skimage
from skimage.feature import local_binary_pattern
from skimage.feature import hog

type vector = np.ndarray

def root_con(*args) -> vector:
    assert not any([a == None for a in args])
    feature_vector = np.concatenate((args),axis=0)
    return feature_vector

def conVector(img) -> vector:
    try:
        img_vector=np.concatenate((img))
    except ValueError:
        img_vector=img
    return img_vector

def root_conVector2(img1, img2) -> vector:
    assert not any([a == None for a in [img1, img2]])

    image1=conVector(img1)
    image2=conVector(img2)
    feature_vector=np.concatenate((image1, image2),axis=0)
    assert feature_vector is not None
    return feature_vector

def root_conVector3(img1, img2, img3) -> vector:
    image1=conVector(img1)
    image2=conVector(img2)
    image3=conVector(img3)
    feature_vector=np.concatenate((image1, image2, image3),axis=0)
    return feature_vector


def root_con_vector(*images) -> vector:
    assert all(img is not None for img in images)
    images = tuple(conVector(img) for img in images)

    return np.concatenate(images, axis=0)


def all_lbp(image) -> vector:
    #uniform_LBP
    lbp=local_binary_pattern((image*255).astype(np.uint8), P=8, R=1.5, method='nri_uniform')
    n_bins = 59
    hist, _ =np.histogram(lbp,n_bins, (0,59))
    return hist

def HoGFeatures(image) -> vector:
    try:
        _, realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                    transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image

def hog_features_patches(image,patch_size,moving_size):
    img=np.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = np.zeros((len(patch)))
    realImage=HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = np.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    return hog_features

def global_hog_small(image):
    feature_vector = hog_features_patches(image, 4, 4)
    return feature_vector

def all_sift(image):
    width,height=image.shape
    min_length=np.min((width,height))
    img=np.asarray(image[0:width,0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length,0:min_length])
    # dimension 128 for all images
    w,h=feaArrSingle.shape
    feature_vector=np.reshape(feaArrSingle, (h,))
    return feature_vector

def gau(left, si):
    return gaussian(left,sigma=si)

def gauD(left, si, or1, or2):
    return ndimage.gaussian_filter(left,sigma=si, order=[or1,or2]) # type: ignore Also takes in a sequence of ints

def gab(left,the,fre):
    fmax = np.pi/2
    a = np.sqrt(2)
    freq = fmax/(a**fre)
    thea = np.pi*the/8
    filt_real, _ = np.asarray(gabor(left,theta=thea,frequency=freq))
    return filt_real

def laplace(left):
    return ndimage.laplace(left)

def gaussian_Laplace1(left):
    return ndimage.gaussian_laplace(left,sigma=1)

def gaussian_Laplace2(left):
    return ndimage.gaussian_laplace(left,sigma=2)

def sobelxy(left):
    left=sobel(left)
    return left

def sobelx(left):
    left=ndimage.sobel(left,axis=0)
    return left

def sobely(left):
    left=ndimage.sobel(left,axis=1)
    return left

#max filter
def maxf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.maximum_filter(x,size)
    return x

#median_filter
def medianf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.median_filter(x,size)
    return x

#mean_filter
def meanf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.convolve(x, np.full((3, 3), 1 / (size * size)))
    return x

#minimum_filter
def minf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x=ndimage.minimum_filter(x,size)
    return x

def lbp(image):
    # 'uniform','default','ror','var'
    try:
        lbp = local_binary_pattern((image*255).astype(np.uint8), 8, 1.5, method='nri_uniform')
        lbp = np.divide(lbp,59)
    except: lbp = image
    return lbp


def hog_feature(image):
    try:
        img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                            transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image

def mis_match(img1,img2):
    w1,h1=img1.shape
    w2,h2=img2.shape
    w=min(w1,w2)
    h=min(h1,h2)
    return img1[0:w,0:h],img2[0:w,0:h]

def weighted_add(img1, w1, img2, w2):
    assert all(e is not None for e in (img1, w1, img2, w2))
    cropped_img1, cropped_img2 = mis_match(img1,img2)
    return w1*cropped_img1 + w2*cropped_img2

def weighted_sub(img1, w1, img2, w2):
    assert all(e is not None for e in (img1, w1, img2, w2))
    cropped_img1, cropped_img2 = mis_match(img1,img2)
    return w1*cropped_img1 - w2*cropped_img2

def sqrt(left):
    assert left is not None
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.sqrt(left,)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1

    assert x is not None
    return x

def relu(left):
    return (abs(left)+left)/2

def maxP(left, kel1, kel2):
    assert all([e is not None for e in [left, kel1, kel2]])
    try:
        current = skimage.measure.block_reduce(left, (kel1, kel2), np.max)
    except ValueError:
        current=left

    assert current is not None
    return current

def ZeromaxP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),np.max)
    except ValueError:
        current=left
    zero_p=addZerosPad(left,current)
    return zero_p

def addZerosPad(final, current):
    M, N = final.shape
    m1, n1 = current.shape
    pUpperSize = (M-m1) // 2
    pDownSize = M - pUpperSize - m1
    pLeftSize = (N-n1) // 2
    pRightSize = N - pLeftSize - n1
    PUpper = np.zeros((pUpperSize,n1))
    PDown = np.zeros((pDownSize,n1))
    current = np.concatenate((PUpper,current,PDown),axis=0)
    m2, n2 = current.shape
    PLeft = np.zeros((m2,pLeftSize))
    PRight = np.zeros((m2,pRightSize))
    current = np.concatenate((PLeft,current,PRight),axis=1)
    return current

def conv_filters(image, filters):
    assert all(e is not None for e in [image, filters])
    # length = len(filters)
    # print(filters, len(filters))
    # size = int(sqrt(length))
    # filters_resize = np.asarray(filters).reshape(size, size)
    img = ndimage.convolve(image, np.array(filters))
    assert img is not None
    return img
