import numpy as np
import random
BASE_PATH = "../../datasets"
# READ_PATH = f"{BASE_PATH}/catagory_balanced_datasets"
READ_PATH = f"{BASE_PATH}/even_datasets"
# READ_PATH = f"{BASE_PATH}/JAFFE"

x_train = np.load(f"{READ_PATH}/x_train.npy")
y_train = np.load(f"{READ_PATH}/y_train.npy")
x_validation = np.load(f"{READ_PATH}/x_val.npy")
y_validation = np.load(f"{READ_PATH}/y_val.npy")
x_test = np.load(f"{READ_PATH}/x_test.npy")
y_test = np.load(f"{READ_PATH}/y_test.npy")
