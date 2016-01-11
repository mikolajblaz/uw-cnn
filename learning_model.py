import numpy

import theano
import theano.tensor as T

from cnn_layers import LeNetConvPoolLayer, LogisticRegression, HiddenLayer, relu


class ConvolutionalNeuralNetwork:
    def __init__(self, rng, input, nkerns, batch_size, input_shape):
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

        def recalculate_input_shape():
            input_shape[0] = ((input_shape[0] - filter_shape[0]) / conv_stride[0] + 1) / poolsize[0]
            input_shape[1] = ((input_shape[1] - filter_shape[1]) / conv_stride[1] + 1) / poolsize[1]

        set_layer_parameters((3, 3))

        # First convolutional pooling layer
        layer0 = LeNetConvPoolLayer(
            rng,
            input=input,
            image_shape=(batch_size, 3, input_shape[0], input_shape[1]),
            filter_shape=(nkerns[0], 3, filter_shape[0], filter_shape[1]),
            poolsize=poolsize,
            conv_stride=conv_stride,
            activation=relu
        )

        recalculate_input_shape()
        set_layer_parameters((3, 3), st=(1, 1))

        # Second convolutional pooling layer
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], input_shape[0], input_shape[1]),
            filter_shape=(nkerns[1], nkerns[0], filter_shape[0], filter_shape[1]),
            poolsize=poolsize,
            conv_stride=conv_stride,
            activation=relu
        )

        recalculate_input_shape()

        # Prepare input for a fully connected layer
        layer2_input = layer1.output.flatten(2)

        # Fully connected layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * input_shape[0] * input_shape[1],
            n_out=nkerns[2],
            activation=relu
        )

        # classification
        layer3 = LogisticRegression(input=layer2.output, n_in=nkerns[2], n_out=nkerns[3])

        output_layer = layer3

        # INTERFACE
        self.params = layer3.params + layer2.params + layer1.params + layer0.params
        self.negative_log_likelihood = output_layer.negative_log_likelihood
        self.errors = output_layer.errors
        self.input = input

        # TODO: maybe delegate L1/L2 counting to layers?
        # L1 regularization
        self.L1 = (abs(layer0.W).sum() + abs(layer1.W).sum() + abs(layer2.W).sum() + abs(layer3.W).sum())

        # L2 regularization
        self.L2 = ((layer0.W ** 2).sum() + (layer1.W ** 2).sum() + (layer2.W ** 2).sum() + (layer3.W ** 2).sum())

        # outputs
        self.predict = output_layer.y_pred
        self.output = output_layer.p_y_given_x


class LearningModel:
    def __init__(self, rng, datasets, nkerns=(20, 50, 50, 10), batch_size=10, input_shape=(28, 28),
                 learning_rate=0.01, L1_reg=0.1, L2_reg=0.1):
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        self.batch_size = batch_size
        self.n_train_batches = n_train_batches
        self.n_valid_batches = n_valid_batches
        self.n_test_batches = n_test_batches

        # symbolic variables
        index = T.lscalar()  # index to a [mini]batch
        x = T.tensor4('x')  # input
        y = T.ivector('y')  # labels

        classifier = ConvolutionalNeuralNetwork(rng, input=x, nkerns=nkerns,
                                                batch_size=batch_size, input_shape=input_shape)

        cost = (classifier.negative_log_likelihood(y) +
                L1_reg * classifier.L1 +
                L2_reg * classifier.L2)

        # functions computing mistakes
        self.test_model = theano.function(
            [index],
            classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            [index],
            classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        params = classifier.params
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - learning_rate * grad_i)
                   for param_i, grad_i in zip(params, grads)]

        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # outputs
        self.predict_model = theano.function(
            [index],
            classifier.predict,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.output_model = theano.function(
            [index],
            classifier.output,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        # parameters printing function
        print_op = theano.printing.Print('params')
        params_values = [print_op(param) for param in params]
        self.display_params = theano.function([], params_values)

    def train(self, n_epochs, verbose=False):
        # TODO: save intermediate results to a file
        batch_size = self.batch_size
        n_train_batches = self.n_train_batches
        n_valid_batches = self.n_valid_batches
        n_test_batches = self.n_test_batches

        validation_frequency = n_train_batches

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.

        epoch = 0

        while epoch < n_epochs:
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                cost_ij = self.train_model(minibatch_index)
                if verbose:
                    print 'training @ iter = ', iter, 'cost = ', cost_ij

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in xrange(n_valid_batches)]
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
                        test_losses = [
                            self.test_model(i)
                            for i in xrange(n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        if verbose:
                            print(('     epoch %i, minibatch %i/%i, test error of '
                                   'best model %f %%') %
                                  (epoch, minibatch_index + 1, n_train_batches,
                                   test_score * 100.))

        return best_validation_loss, best_iter, test_score

    def predict(self):
        for i in range(self.n_train_batches):
            print self.predict_model(i)
