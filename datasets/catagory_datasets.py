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


BASE_PATH = "/home/loaf/Downloads/train_set"
WRITE_PATH = "catagory_dataset"
# WRITE_PATH = "happy_sad_dataset"

def img_from_num(number) -> np.ndarray:
    return np.array(Image.open(f"{BASE_PATH}/images/{number}.jpg").convert("L"))

def arousal_valence_from_number(number) -> tuple[float, float]:
    return (
        float(np.load(f"{BASE_PATH}/annotations/{number}_aro.npy")),
        float(np.load(f"{BASE_PATH}/annotations/{number}_val.npy"))
    )

def expression_index_from_number(number) -> int:
    return int(np.load(f"{BASE_PATH}/annotations/{number}_exp.npy"))


def save_x_y(start: int, end: int, name: str, samples: list[list[int]]) -> None:

        sample_subset = [catagory[start:end] for catagory in samples]

        flat_samples: list[int] = sum(sample_subset, [])
        random.shuffle(flat_samples)

        np.save(f"{WRITE_PATH}/x_{name}.npy", np.array([img_from_num(number) for number in flat_samples]))
        np.save(f"{WRITE_PATH}/y_{name}.npy", np.array([expression_index_from_number(number) for number in flat_samples]))


def make_datasets() -> None:
    #0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt
    NUM_CATAGORIES = 8
    catagories: list[list[int]] = [[] for _ in range(NUM_CATAGORIES)]

    num_files = len(os.listdir(f"{BASE_PATH}/images"))
    for i, name in enumerate(os.listdir(f"{BASE_PATH}/images")):
        if i % 1000 == 0:
            print(f"{i}/{num_files}: {((i/num_files)*100):.1f}%")

        number = int(name[:-4])

        exp_index = expression_index_from_number(number)

        catagories[exp_index].append(number)

    for catagory in catagories:
        random.shuffle(catagory)


    start = 0
    filtered_catagories = [catagory for i, catagory in enumerate(catagories)] #if i in (1, 2) ]
    for name, size in [("train", 1_000), ("test", 800), ("val", 500)]:
        size //= len(filtered_catagories)
        print("saving: "+name)
        save_x_y(start, start+size, name, filtered_catagories)
        start += size




if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    make_datasets()

