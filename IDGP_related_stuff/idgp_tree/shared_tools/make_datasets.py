import numpy as np
BASE_PATH = "../../datasets"
# READ_PATH = f"{BASE_PATH}/catagory_balanced_datasets"
READ_PATH = f"{BASE_PATH}/even_datasets"
#READ_PATH = f"{BASE_PATH}/JAFFE"

x_train = np.load(f"{READ_PATH}/x_train.npy") / 255
y_train = np.load(f"{READ_PATH}/y_train.npy")
x_validation = np.load(f"{READ_PATH}/x_val.npy") / 255
y_validation = np.load(f"{READ_PATH}/y_val.npy")
x_test = np.load(f"{READ_PATH}/x_test.npy") / 255
y_test = np.load(f"{READ_PATH}/y_test.npy")


if __name__ == '__main__':
    print("x_train: ", len(x_train))
    print("y_train: ", len(y_train))
    print("x_val: ", len(x_validation))
    print("y_val: ", len(y_validation))
    print("x_test: ", len(x_test))
    print("y_test: ", len(y_test))
