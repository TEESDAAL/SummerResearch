import random, os, numpy as np
from typing import TypeVar
from PIL import Image


T, A, B = TypeVar('T'), TypeVar('A'), TypeVar('B')

type Img = np.ndarray

READ_PATH = "/home/loaf/Downloads/jaffedbase/jaffedbase"
WRITE_PATH = "../datasets"



def make_datasets() -> None:
    catagories: dict[str, list[str]] = {}
    for i, name in enumerate(os.listdir(READ_PATH)):
        if 'txt' in name:
            continue

        emotion = name.split('.')[1][:2]
        catagories[emotion] = catagories.get(emotion, []) + [name]

    surprised: list[tuple[Img, int]] = [(to_img(file_name), 1) for file_name in catagories['SU']]
    happy: list[tuple[Img, int]] = [(to_img(file_name), -1) for file_name in catagories['HA']]
    h_train, h_test = test_train_split(happy)
    h_train, h_val = test_train_split(h_train)

    s_train, s_test = test_train_split(surprised)
    s_train, s_val = test_train_split(s_train)

    x_train, y_train = unzip(shuffle(h_train + s_train))
    x_val, y_val = unzip(shuffle(h_train + s_train))
    x_test, y_test = unzip(shuffle(h_test + s_test))
    x_val, y_val = unzip(shuffle(h_val + s_val))

    for x, y, set_type in [(x_train, y_train, 'train'),
                           (x_val, y_val, 'val'),(x_test, y_test, 'test')]:

        np.save(f"{WRITE_PATH}/x_{set_type}.npy", x)
        np.save(f"{WRITE_PATH}/y_{set_type}.npy", y)



def to_img(file_name: str) -> Img:
    return np.array(Image.open(f"{READ_PATH}/{file_name}").convert("L"))


def unzip(lst: list[tuple[A, B]]) -> tuple[list[A], list[B]]:
    return tuple(zip(*lst))

def shuffle(lst: list[T]) -> list[T]:
    return random.sample(lst, k=len(lst))

def test_train_split(dataset: list[T], train_size: float = 0.8) -> tuple[list[T], list[T]]:
    dataset = shuffle(dataset)
    split_index = int(train_size*len(dataset))
    return dataset[:split_index], dataset[split_index:]


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
