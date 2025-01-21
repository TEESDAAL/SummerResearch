import numpy as np
from skimage.transform import rescale
from functools import partial

BASE_PATH = "../datasets"
#READ_PATH = f"{BASE_PATH}/catagory_balanced_datasets"
#READ_PATH = f"{BASE_PATH}/even_datasets"
#READ_PATH = f"{BASE_PATH}/JAFFE"
READ_PATH = f"{BASE_PATH}/happy_sad_dataset"

x_train = np.array([rescale(img, 0.5) for img in np.load(f"{READ_PATH}/x_train.npy") / 255])
y_train = np.load(f"{READ_PATH}/y_train.npy")
x_validation =  np.array([rescale(img, 0.5) for img in np.load(f"{READ_PATH}/x_val.npy") / 255])
y_validation = np.load(f"{READ_PATH}/y_val.npy")
x_test = np.array([rescale(img, 0.5) for img in np.load(f"{READ_PATH}/x_test.npy") / 255])
y_test = np.load(f"{READ_PATH}/y_test.npy")
print("Loaded datasets")

