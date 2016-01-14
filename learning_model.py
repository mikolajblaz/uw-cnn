import numpy

import theano
import theano.tensor as T

from cnn_layers import Conv, Pool, FC, Softmax, relu
from augmentation import Augmentation


class ConvolutionalNeuralNetwork:
    """
    Network architecture:
    Conv -> Relu -> Pool -> FC -> Softmax
    """
    def __init__(self, rng, input, n_units, batch_size, input_shape):
        """
        :type n_units: list
        :param n_units: for Conv layers it means depth, for FC and softmax - number of hidden units
        :param augmentation: an object allowing random data augmentation
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
            image_shape=(batch_size, 1, input_shape[0], input_shape[1]),
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
    def __init__(self, rng, nkerns=(20, 50, 10), batch_size=10, input_shape=(28, 28), cropped_input_shape=(26, 26),
                 learning_rate=0.01, L1_reg=0.1, L2_reg=0.1):

        self.batch_size = batch_size
        self.augm = Augmentation(rng, input_shape, cropped_input_shape, batch_size)

        # symbolic variables
        x = T.tensor4('x')  # input
        y = T.ivector('y')  # labels

        #augmented_input = augm.augment_batch(input, (28, 28), (26, 26), batch_size)

        classifier = ConvolutionalNeuralNetwork(rng, input=x, n_units=nkerns,
                                                batch_size=batch_size, input_shape=cropped_input_shape)

        cost = (classifier.negative_log_likelihood(y) +
                L1_reg * classifier.L1 +
                L2_reg * classifier.L2)

        # functions computing mistakes
        self.test_model = theano.function(
            [x, y],
            classifier.errors(y)
        )

        self.validate_model = theano.function(
            [x, y],
            classifier.errors(y)
        )

        params = classifier.params
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - learning_rate * grad_i)
                   for param_i, grad_i in zip(params, grads)]

        self.train_model = theano.function(
            [x, y],
            cost,
            updates=updates
        )
        #
        # self.train_model_errors = theano.function(
        #     [index],
        #     classifier.errors(y),
        #     givens={
        #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
        #         y: train_set_y[index * batch_size: (index + 1) * batch_size]
        #     }
        # )
        #
        # # outputs
        # self.predict_model = theano.function(
        #     [index],
        #     classifier.predict,
        #     givens={
        #         x: train_set_x[index * batch_size: (index + 1) * batch_size]
        #     }
        # )
        #
        # self.output_model = theano.function(
        #     [index],
        #     classifier.output,
        #     givens={
        #         x: train_set_x[index * batch_size: (index + 1) * batch_size]
        #     }
        # )

        # parameters printing function
        print_op = theano.printing.Print('params')
        params_values = [print_op(param) for param in params]
        self.display_params = theano.function([], params_values)

    def train(self, datasets, n_epochs, verbose=False):
        # TODO: save intermediate results to a file
        batch_size = self.batch_size

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_train_batches = train_set_x.shape[0]
        n_valid_batches = valid_set_x.shape[0]
        n_test_batches = test_set_x.shape[0]
        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        validation_frequency = n_train_batches

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.

        epoch = 0

        augm = self.augm

        while epoch < n_epochs:
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                cost_ij = self.train_model(augm.augment_batch(train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]),
                                           train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])

                if verbose:
                    print 'training @ iter = ', iter, 'cost = ', cost_ij

                if (iter + 1) % validation_frequency == 0:

                    # train_losses = [self.train_model_errors(i) for i
                    #                      in xrange(n_train_batches)]
                    # this_train_loss = numpy.mean(train_losses)
                    # if verbose:
                    #     print('epoch %i, minibatch %i/%i, train error %f %%' %
                    #           (epoch, minibatch_index + 1, n_train_batches,
                    #            this_train_loss * 100.))

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(self.augm.augment_batch(valid_set_x[i * batch_size: (i + 1) * batch_size]),
                                                             valid_set_y[i * batch_size: (i + 1) * batch_size])
                                        for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    if verbose:
                        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                              (epoch, minibatch_index + 1, n_train_batches,
                               this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [self.test_model(self.augm.augment_batch(test_set_x[i * batch_size: (i + 1) * batch_size]),
                                                       test_set_y[i * batch_size: (i + 1) * batch_size])
                                       for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        if verbose:
                            print(('     epoch %i, minibatch %i/%i, test error of '
                                   'best model %f %%') %
                                  (epoch, minibatch_index + 1, n_train_batches,
                                   test_score * 100.))

        return best_validation_loss, best_iter, test_score

    # def predict(self):
    #     return [self.predict_model(i) for i in range(n_train_batches)]
