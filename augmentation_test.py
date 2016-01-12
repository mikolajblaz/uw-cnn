import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from augmentation import functions


def show(image):
    imgplot = plt.imshow(image)

test_image = None

for fun in functions:
    show(fun(test_image))
