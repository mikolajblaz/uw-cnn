import numpy

import data_preparation as dp
from learning_model import LearningModel


def basic_nn_input():
    batch_size = 10
    rng = numpy.random.RandomState(123456)
    dataset = dp.load_data('mnist.pkl.gz')
    model = LearningModel(rng, dataset, batch_size=batch_size)

    print model.train(20, verbose=True)

    print 'Predicted labels after training:'
    model.predict()


def basic_flow_test():
    basic_nn_input()


if __name__ == '__main__':
    basic_flow_test()
