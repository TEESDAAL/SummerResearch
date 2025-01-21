import random
from PIL import Image
import os
import numpy as np
import functools
import random
from PIL import Image
import os
import numpy as np
import functools

BASE_PATH = "../train_set"

def img_from_num(number) -> np.ndarray:
    return np.array(Image.open(f"{BASE_PATH}/images/{number}.jpg").convert("L"))

def arousal_valence_from_number(number) -> tuple[float, float]:
    return (
        float(np.load(f"{BASE_PATH}/annotations/{number}_aro.npy")),
        float(np.load(f"{BASE_PATH}/annotations/{number}_val.npy"))
    )


def save_x_y(start: int, end: int, name: str, samples: list[list[str]]) -> None:
        sample_subset = [catagory[start:end] for catagory in samples]

        flat_samples: list[str]  = functools.reduce(lambda list1, list2: list1 + list2, sample_subset)
        random.shuffle(samples)

        np.save(f"x_{name}.npy", np.array([img_from_num(number) for number in flat_samples]))
        np.save(f"y_{name}.npy", np.array([arousal_valence_from_number(number) for number in flat_samples]))


def make_datasets() -> None:
    NUM_CATAGORIES = 8
    catagories: list[list[str]] = [[] for _ in range(NUM_CATAGORIES)]

    num_files = len(os.listdir(f"{BASE_PATH}/images"))
    for i, name in enumerate(os.listdir(f"{BASE_PATH}/images")):
        if i % 1000 == 0:
            print(f"{i}/{num_files}: {((i/num_files)*100):.1f}%")

        number = name[:-4]
        exp_index = int(np.load(f"{BASE_PATH}/annotations/{number}_exp.npy"))
        catagories[exp_index].append(number)

    for catagory in catagories:
        random.shuffle(catagory)

    start = 0
    for name, size in [("train", 100), ("test", 100), ("val", 50)]:
        print("saving: "+name)
        save_x_y(start, start+size, name, catagories)
        start += size




if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    make_datasets()


x_train = np.load("datasets/x_train.npy")
y_train = np.load("datasets/y_train.npy")
x_validation = np.load("datasets/x_val.npy")
y_validation = np.load("datasets/y_val.npy")
x_test = np.load("datasets/x_test.npy")
y_test = np.load("datasets/y_test.npy")
