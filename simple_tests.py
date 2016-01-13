import numpy

import data_preparation as dp
from learning_model import LearningModel


def basic_nn_input():
    rng = numpy.random.RandomState(123456)
    dataset = dp.load_data('mnist.pkl.gz')
    model = LearningModel(rng, dataset, nkerns=(20, 50, 10), batch_size=500, input_shape=(28, 28),
                          learning_rate=0.01, L1_reg=0.001, L2_reg=0.001)

    print model.train(10, verbose=True)

    # print 'Predicted labels after training:'
    # pred = model.predict()
    # print pred


def basic_flow_test():
    basic_nn_input()


if __name__ == '__main__':
    basic_flow_test()
