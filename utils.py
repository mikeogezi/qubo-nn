import numpy as np


def make_dataset():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
    t = np.array([[0], [1], [1], [0]], "float32")
    return x, t
