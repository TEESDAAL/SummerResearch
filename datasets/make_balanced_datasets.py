import random, math, os, numpy as np, random, itertools, numpy.typing as npt
from PIL import Image
from typing import TypeVar

A, B = TypeVar('A'), TypeVar('B')

BASE_PATH = "/home/loaf/Downloads/train_set"
WRITE_PATH = "even_datasets"
ImgNum = int | str
ValAroPair = tuple[np.floating, np.floating] | tuple[float, float]

def make_datasets() -> None:
    catagories: dict[ValAroPair, list[ImgNum]] = {
        p: [] for p in itertools.product(points_in_range(-1, 1, n=10), repeat=2)
    }

    num_files = len(os.listdir(f"{BASE_PATH}/images"))
    for i, name in enumerate(os.listdir(f"{BASE_PATH}/images")):
        if i % 1000 == 0:
            print(f"{i}/{num_files}: {((i/num_files)*100):.1f}%")

        number = int(name[:-4])
        a, v = arousal_valence_from_number(number)
        closest_cell = min(catagories, key=lambda p: math.hypot(a-p[0], v-p[1]))
        catagories[closest_cell].append(number)

    for cell in catagories:
        random.shuffle(catagories[cell])

    train_set = []
    val_set = []
    test_set = []
    get_cell = itertools.cycle(catagories.values())

    populate_set(train_set, 1_000, get_cell)
    populate_set(val_set, 500, get_cell)
    populate_set(test_set, 800, get_cell)

    for numbers, name in [(train_set, 'train'), (val_set, 'val'), (test_set, 'test')]:
        np.save(f"{WRITE_PATH}/x_{name}.npy", np.array([img_from_num(n) for n in numbers]))
        np.save(f"{WRITE_PATH}/y_{name}.npy", np.array([arousal_valence_from_number(n) for n in numbers]))



def img_from_num(number: ImgNum) -> npt.NDArray[np.floating]:
    return np.array(Image.open(f"{BASE_PATH}/images/{number}.jpg").convert("L"))


def arousal_valence_from_number(number: ImgNum) -> tuple[float, float]:
    return (
        float(np.load(f"{BASE_PATH}/annotations/{number}_aro.npy")),
        float(np.load(f"{BASE_PATH}/annotations/{number}_val.npy"))
    )


def points_in_range(start: float, end: float, n: float) -> npt.NDArray[np.floating]:
    step = (end - start)/(n-1)
    return np.arange(start, end + step, step)

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
