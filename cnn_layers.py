import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


def relu(x):
    return T.switch(x > 0, x, 0)


class Softmax:
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W is None:
            W_bound = numpy.sqrt(6. / (n_in + n_out))
            W_values = numpy.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.input = input

        # outputs in range 0 to 1
        self.y = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # output class
        self.y_pred = T.argmax(self.y, axis=1)

    def nll(self, y):
        return -T.mean(T.log(self.y)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        :param y: vactor of correct labels (of type int_)
        :return: mean of errors
        """
        # TODO: maybe remove check?
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class FC:
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh, W=None, b=None):
        if W is None:
            W_bound = numpy.sqrt(6. / (n_in + n_out))
            W_values = numpy.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.input = input

        lin_output = T.dot(input, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)


class Conv:
    def __init__(self, rng, input, filter_shape=None, image_shape=None, conv_stride=(1, 1), activation=T.tanh,
                 W=None, b=None):
        assert image_shape[1] == filter_shape[1]

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])

        if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            W_values = numpy.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=filter_shape
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            # one bias for each feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.input = input

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=conv_stride
        )

        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = lin_output if activation is None else activation(lin_output)


class Pool:
    def __init__(self, input, poolsize=(2, 2)):
        self.input = input

        self.output = downsample.max_pool_2d(
            input=input,
            ds=poolsize,
            ignore_border=True
        )
