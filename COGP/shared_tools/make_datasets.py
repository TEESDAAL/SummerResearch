import numpy as np
from skimage.transform import resize
BASE_PATH = "../datasets"
# READ_PATH = f"{BASE_PATH}/catagory_balanced_datasets"
READ_PATH = f"{BASE_PATH}/even_datasets"
#READ_PATH = f"{BASE_PATH}/JAFFE"
downscale = lambda img: resize(img, (64, 64))
x_train = np.array([downscale(img) for img in np.load(f"{READ_PATH}/x_train.npy") / 255])
y_train = np.load(f"{READ_PATH}/y_train.npy")
x_validation = np.array([downscale(img) for img in np.load(f"{READ_PATH}/x_val.npy") / 255])
y_validation = np.load(f"{READ_PATH}/y_val.npy")
x_test = np.array([downscale(img) for img in np.load(f"{READ_PATH}/x_test.npy") / 255])
y_test = np.load(f"{READ_PATH}/y_test.npy")

