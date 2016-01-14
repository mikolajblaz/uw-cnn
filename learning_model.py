import numpy

import theano
import theano.tensor as T

from cnn_layers import Conv, Pool, FC, Softmax, relu


class ConvolutionalNeuralNetwork:
    """
    Network architecture:
    Conv -> Relu -> Pool -> FC -> Softmax
    """
    def __init__(self, rng, input, n_units, input_shape):
        """
        :type n_units: list
        :param n_units: for Conv layers it means depth, for FC and softmax - number of hidden units
        """
        input_shape = list(input_shape)
        filter_shape = [5, 5]
        poolsize = [2, 2]
        conv_stride = [1, 1]

        def set_layer_parameters(filter_sh=None, pool=None, st=None):
            if filter_sh is not None:
                filter_shape[0] = filter_sh[0]
                filter_shape[1] = filter_sh[1]
            if pool is not None:
                poolsize[0] = pool[0]
                poolsize[1] = pool[1]
            if st is not None:
                conv_stride[0] = st[0]
                conv_stride[1] = st[1]

        def recalculate_after_conv():
            input_shape[0] = (input_shape[0] - filter_shape[0]) / conv_stride[0] + 1
            input_shape[1] = (input_shape[1] - filter_shape[1]) / conv_stride[1] + 1

        def recalculate_after_pool():
            input_shape[0] /= poolsize[0]
            input_shape[1] /= poolsize[1]

        # Conv
        set_layer_parameters((3, 3))
        layer0 = Conv(
            rng,
            input=input,
            image_shape=(None, 1, input_shape[0], input_shape[1]),
            filter_shape=(n_units[0], 1, filter_shape[0], filter_shape[1]),
            conv_stride=conv_stride,
            activation=relu
        )
        recalculate_after_conv()

        # Pool
        set_layer_parameters((3, 3), st=(1, 1))
        pool_player = Pool(
            input=layer0.output,
            poolsize=poolsize
        )
        recalculate_after_pool()

        # Prepare input for a fully connected layer
        layer1_input = pool_player.output.flatten(2)

        # FC
        layer1 = FC(
            rng,
            input=layer1_input,
            n_in=n_units[0] * input_shape[0] * input_shape[1],
            n_out=n_units[1],
            activation=relu
        )

        # classification
        layer2 = Softmax(rng=rng, input=layer1.output, n_in=n_units[1], n_out=n_units[2])

        output_layer = layer2

        # INTERFACE
        self.params = layer2.params + layer1.params + layer0.params
        self.negative_log_likelihood = output_layer.nll
        self.errors = output_layer.errors
        self.input = input

        # TODO: maybe delegate L1/L2 counting to layers?
        # L1 regularization
        self.L1 = (abs(layer0.W).sum() + abs(layer1.W).sum() + abs(layer2.W).sum())

        # L2 regularization
        self.L2 = ((layer0.W ** 2).sum() + (layer1.W ** 2).sum() + (layer2.W ** 2).sum())

        # outputs
        self.predict = output_layer.y_pred
        self.output = output_layer.y


class LearningModel:
    def __init__(self, rng, nkerns=(20, 50, 10), input_shape=(26, 26), learning_rate=0.01, L1_reg=0.1, L2_reg=0.1):
        # symbolic variables
        x = T.tensor4('x')  # input
        y = T.ivector('y')  # labels

        classifier = ConvolutionalNeuralNetwork(rng, input=x, n_units=nkerns, input_shape=input_shape)

        cost = (classifier.negative_log_likelihood(y) +
                L1_reg * classifier.L1 +
                L2_reg * classifier.L2)

        params = classifier.params
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - learning_rate * grad_i)
                   for param_i, grad_i in zip(params, grads)]

        self.train_model = theano.function([x, y], cost, updates=updates)

        # outputs
        self.errors = theano.function([x, y], classifier.errors(y))
        self.predict = theano.function([x], classifier.predict)
        self.output = theano.function([x], classifier.output)

        # parameters printing function
        print_op = theano.printing.Print('params')
        params_values = [print_op(param) for param in params]
        self.display_params = theano.function([], params_values)

    def train(self, datasets, batch_size, n_epochs, image_processing, verbose=False):
        """
        :param image_processing: an object allowing random data augmentation and processing
        """
        # TODO: save intermediate results to a file
        proc = image_processing

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # Crop static sets
        valid_set_x = proc.augment_batch(valid_set_x, random=False)
        test_set_x = proc.augment_batch(test_set_x, random=False)

        n_train = train_set_x.shape[0]
        n_valid = valid_set_x.shape[0]
        n_test = test_set_x.shape[0]

        validation_frequency = 1

        best_validation_loss = numpy.inf
        best_epoch = 0

        epoch = 0
        iter = 0

        while epoch < n_epochs:
            epoch += 1
            for idx_l, idx_p in zip(range(0, n_train, batch_size), range(batch_size, n_train, batch_size)):
                iter += 1
                cost_ij = self.train_model(proc.augment_batch(train_set_x[idx_l: idx_p]), train_set_y[idx_l: idx_p])

                if verbose:
                    print 'training @ iter = ', iter, 'cost = ', cost_ij

            if epoch % validation_frequency == 0:
                train_loss = self.errors(proc.augment_batch(train_set_x, random=False), train_set_y)
                if verbose:
                    print 'epoch %i: train error %f %%' % (epoch, train_loss * 100.)

                valid_loss = self.errors(valid_set_x, valid_set_y)
                print 'epoch %i: validation error %f %%' % (epoch, valid_loss * 100.)

                if valid_loss < best_validation_loss:
                    best_validation_loss = valid_loss
                    best_epoch = epoch
                    test_loss = self.errors(test_set_x, test_set_y)
                    print '    best model with test error %f %%' % (test_loss * 100.)
