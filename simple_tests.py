import numpy

import data_preparation as dp
from learning_model import LearningModel


def basic_nn_input():
    rng = numpy.random.RandomState(123456)
    img_proc = dp.ImageProcessing(rng, image_shape=(28, 28), cropped_image_shape=(26, 26))
    model = LearningModel(rng, nkerns=(2, 5, 10), input_shape=(26, 26),
                          learning_rate=0.01, L1_reg=0.001, L2_reg=0.001)

    dataset = dp.load_mnist()
    for set in dataset:
        print set[0].shape[0]
    model.train(dataset, n_epochs=10, batch_size=500, image_processing=img_proc, verbose=False)

    # print 'Predicted labels after training:'
    # pred = model.predict()
    # print pred


if __name__ == '__main__':
    basic_nn_input()
