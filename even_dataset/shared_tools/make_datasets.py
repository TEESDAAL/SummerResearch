import random, math, os, numpy as np, random, itertools
from PIL import Image
from typing import TypeVar

A, B = TypeVar('A'), TypeVar('B')

BASE_PATH = "/home/loaf/Downloads/train_set"
WRITE_PATH = "../datasets"
ImgNum = int | str


def img_from_num(number: ImgNum) -> np.ndarray:
    return np.array(Image.open(f"{BASE_PATH}/images/{number}.jpg").convert("L"))

def arousal_valence_from_number(number: ImgNum) -> tuple[float, float]:
    return (
        float(np.load(f"{BASE_PATH}/annotations/{number}_aro.npy")),
        float(np.load(f"{BASE_PATH}/annotations/{number}_val.npy"))
    )

def points_in_range(start, end, n):
    step = (end - start)/(n-1)
    return np.arange(start, end + step, step)


def make_datasets() -> None:
    catagories: dict[tuple[float, float], list[tuple[ImgNum, tuple[float, float]]]] = {p: [] for p in itertools.product(points_in_range(-1, 1, n=10), repeat=2)}
    num_files = len(os.listdir(f"{BASE_PATH}/images"))
    for i, name in enumerate(os.listdir(f"{BASE_PATH}/images")):
        if i % 1000 == 0:
            print(f"{i}/{num_files}: {((i/num_files)*100):.1f}%")

        number = int(name[:-4])
        a, v = arousal_valence_from_number(number)
        closest_cell = min(catagories, key=lambda p: math.hypot(a-p[0], v-p[1]))
        catagories[closest_cell].append((number, (a, v)))

    for cell in catagories:
        random.shuffle(catagories[cell])

    train_set = []
    val_set = []
    test_set = []
    get_cell = itertools.cycle(catagories.values())

    populate_set(train_set, 1_000, get_cell)
    populate_set(val_set, 500, get_cell)
    populate_set(test_set, 800, get_cell)

    for dataset, name in [(train_set, 'train'), (val_set, 'val'), (test_set, 'test')]:
        x, y = unzip(dataset)
        np.save(f"{WRITE_PATH}/x_{name}.npy", np.array([img_from_num(n) for n in x]))
        np.save(f"{WRITE_PATH}/y_{name}.npy", np.array(y))



def unzip(lst: list[tuple[A, B]]) -> tuple[list[A], list[B]]:
    return ([a for a, _ in lst], [b for _, b in lst])


def save_x_y(samples, name):
    np.save(f"{WRITE_PATH}/x_{name}.npy", np.array([img_from_num(i) for i in samples]))
    np.save(f"{WRITE_PATH}/y_{name}.npy", np.array([img_from_num(i) for i in samples]))


def populate_set(lst: list, desired_size, generator):
    while len(lst) < desired_size:
        cell = next(generator)
        if len(cell) == 0:
            continue
        lst.append(cell.pop())


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    make_datasets()


x_train = np.load("datasets/x_train.npy") / 255
y_train = np.load("datasets/y_train.npy")
x_validation = np.load("datasets/x_val.npy") / 255
y_validation = np.load("datasets/y_val.npy")
x_test = np.load("datasets/x_test.npy") / 255
y_test = np.load("datasets/y_test.npy")
