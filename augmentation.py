import numpy as np
import theano
import theano.tensor as T


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
    # return image[:]


functions = [lighten, rotate]


class Augmentation:
    def __init__(self, rng, image_shape, cropped_image_shape, batch_size, fun_list=functions):
        self.fun_list = fun_list
        self.length = len(fun_list)
        self.rng = rng
        self.image_shape = image_shape
        self.cropped_image_shape = cropped_image_shape
        self.batch_size = batch_size

    def augment_image(self, image):
        rng = self.rng
        rand_fun_idx = rng.randint(0, self.length)
        fun = self.fun_list[rand_fun_idx]
        return fun(image, rng)

    def augment_batch(self, images):
        new_images = np.zeros(
            (self.batch_size, 1, 26, 26),
            dtype=theano.config.floatX
        )

        images = images.reshape((self.batch_size, 1, 28, 28))

        for i in range(self.batch_size):
            new_images[i, :, :, :] = crop_image(images[i, :, :, :], self.rng, self.image_shape, self.cropped_image_shape)

        return new_images
