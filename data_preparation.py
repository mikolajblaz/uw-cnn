import cPickle
import gzip
import os

import numpy as np
import theano
import theano.tensor as T


def load_mnist():
    filename = 'mnist.pkl.gz'
    # Download the MNIST dataset if it is not present
    if not os.path.isfile(filename):
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, filename)

    print '... loading data'

    # Load the dataset
    f = gzip.open(filename, 'rb')
    dataset = cPickle.load(f)
    f.close()

    return [(set[0], set[1].astype('int32')) for set in dataset]


def lighten(image, rng):
    print "Lighten"
    return image


def rotate(image, rng):
    print "Rotate"
    return image


def crop_image(image, rng, image_shape, new_image_shape):
    left = rng.randint(0, image_shape[0] - new_image_shape[0])
    top = rng.randint(0, image_shape[1] - new_image_shape[1])
    return image[:, top: top + new_image_shape[1], left: left + new_image_shape[0]]


functions = [lighten, rotate]


class ImageProcessing:
    def __init__(self, rng, image_shape, cropped_image_shape, fun_list=functions):
        self.fun_list = fun_list
        self.length = len(fun_list)
        self.rng = rng
        self.image_shape = image_shape
        self.cropped_image_shape = cropped_image_shape

    def augment_image(self, image):
        rng = self.rng
        rand_fun_idx = rng.randint(0, self.length)
        fun = self.fun_list[rand_fun_idx]
        return fun(image, rng)

    def augment_batch(self, images):
        batch_size = images.shape[0]

        new_images = np.zeros(
            (batch_size, 1, 26, 26),
            dtype=theano.config.floatX
        )

        images = images.reshape((batch_size, 1, 28, 28))

        for i in range(batch_size):
            new_images[i, :, :, :] = crop_image(images[i, :, :, :], self.rng, self.image_shape, self.cropped_image_shape)

        return new_images
