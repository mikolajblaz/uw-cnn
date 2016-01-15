import numpy

import data_preparation as dp
from learning_model import LearningModel


def basic_nn_input():
    rng = numpy.random.RandomState(123456)

    # an object responsible for image processing and augmentation
    img_proc = dp.ImageProcessing(rng, image_shape=(28, 28), cropped_image_shape=(26, 26))

    # model
    model = LearningModel(rng, img_proc, n_units=(20, 50, 10), input_shape=(26, 26),
                          learning_rate=0.01, L1_reg=0.001, L2_reg=0.001, rmsprop=True)

    # download MNIST dataset to current directory
    dataset = dp.load_mnist()

    # train model with given parameters (and with high verbosity)
    # NOTE: setting n_epochs to e.g. 40 will give lower error rate, but it will obviously take more time
    model.train(dataset, n_epochs=10, batch_size=500, verbose=True)

    # choose 10 random images from test set
    test_set_x, test_set_y = dataset[2]
    idxs = rng.randint(0, test_set_x.shape[0], (10,))
    tested_x = test_set_x[idxs]
    y_pred = model.predict(tested_x)
    tested_y = test_set_y[idxs]

    print 'Predictions:  ', y_pred
    print 'Expectations: ', tested_y


if __name__ == '__main__':
    basic_nn_input()
