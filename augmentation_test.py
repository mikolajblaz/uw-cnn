import matplotlib.pyplot as plt
import numpy as np

from data_preparation import functions

from scipy import misc


def prepare_test_img():
    sample_img = misc.face(gray=True) / 255.
    return np.expand_dims(sample_img, axis=0)


def show(image):
    plt.imshow(image[0], cmap=plt.cm.gray)


if __name__ == '__main__':
    test_img = prepare_test_img()
    rng = np.random.RandomState(1234)

    print [fun.func_name for fun in functions]

    fun_len = len(functions)
    sublot_idx = 0
    for fun in functions:
        sublot_idx += 1
        plt.axis('off')
        plt.subplot(1, fun_len, sublot_idx)
        show(fun(test_img, rng))
    plt.show()