import cPickle
import gzip
import os

import numpy as np
import theano

from scipy import ndimage


def load_mnist():
    """ Load MNIST dataset if not present already.
    :type return: list of 3 pairs
    :return: Train, validation and test set, each one is a pair of images and labels
    """
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

    print 'Data loaded successfully'

    return [(set[0].reshape((set[0].shape[0], 1, 28, 28)),
             set[1].astype('int32'))
            for set in dataset]


###########################################################################
# augmentation functions

def lightness(image, rng, range=0.1):
    scale = rng.random_sample() * 2 * range - range + 1
    return np.clip(image.dot(scale), 0., 1.)


def rotate(image, rng, range=10):
    angle = rng.random_sample() * 2 * range - range
    rot = ndimage.rotate(image[0], int(angle), reshape=False)
    return np.expand_dims(rot, axis=0)


functions = [lightness, rotate]


class ImageProcessing:
    """ Class allowing image cropping and transformations. """
    def __init__(self, rng, image_shape, cropped_image_shape, fun_list=functions):
        self.fun_list = fun_list
        self.length = len(fun_list)
        self.rng = rng
        self.image_shape = image_shape
        self.cropped_image_shape = cropped_image_shape

    def augment_batch(self, images, random=True):
        """
        If random is True, crop and transform image randomly.
        If random is False, crop centered images in the whole batch.

        :return: cropped (and transformed if random is True) batch
        """
        if random:
            batch_size = images.shape[0]
            new_images = np.zeros((batch_size, 1, 26, 26), dtype=theano.config.floatX)
            for i in range(batch_size):
                new_images[i, ...] = self.augment_image(self.crop_image_random(images[i, ...]))

            return new_images

        else:
            return self.crop_batch(images)

    def crop_image_random(self, image):
        shape = self.image_shape
        new_shape = self.cropped_image_shape

        left = self.rng.randint(0, shape[0] - new_shape[0])
        top = self.rng.randint(0, shape[1] - new_shape[1])
        return image[:, top: top + new_shape[1], left: left + new_shape[0]]

    def crop_batch(self, images):
        shape = self.image_shape
        new_shape = self.cropped_image_shape

        left = (shape[0] - new_shape[0]) // 2
        top = (shape[1] - new_shape[1]) // 2
        return images[:, :, top: top + new_shape[1], left: left + new_shape[0]]

    def augment_image(self, image):
        """ Randomly choose a function and apply it to the image. """
        rand_fun_idx = self.rng.randint(0, self.length)
        fun = self.fun_list[rand_fun_idx]
        return fun(image, self.rng)