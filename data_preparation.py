import cPickle
import gzip
import os

import numpy as np
import theano
import theano.tensor as T

from scipy.misc import imread, imresize, imsave
from operator import itemgetter


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        # shared_x = theano.shared(np.asarray(data_x,
        #                                        dtype=theano.config.floatX),
        #                          borrow=borrow)
        # shared_y = theano.shared(np.asarray(data_y,
        #                                        dtype=theano.config.floatX),
        #                          borrow=borrow)
        shared_x = data_x
        shared_y = data_y
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y.astype('int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval



#
#
#
# def load_data(images_dir, rng, train_ratio, validation_ratio):
#     """ Construct datasets in a format applicable to the class 'LearningModel'
#     :param images_dir: a directory containing resized and labeled images
#     :param rng: random number generator used to shuffle datasets
#     :param train_ratio: a fraction of images used in training set
#     :param validation_ratio: a fraction of images used in validation set
#     :return: Tuple of three pairs of shared theano variables
#     """
#     dataset = load_labeled_images(images_dir)
#     train_val_test_data = prepare_dataset(dataset, rng, train_ratio, validation_ratio)
#     print_statistics(train_val_test_data)
#     # TODO: shared variables needed probably onlu for GPU, consider removing it
#     T_shared_dataset = [shared_dataset(dataset) for dataset in train_val_test_data]
#     return T_shared_dataset
#
#
# def load_labeled_images(images_dir):
#     """ Construct two corresponding lists of images and labels. """
#     old_dir = os.getcwd()
#     try:
#         os.chdir(images_dir)
#
#         results = io_module.read_config(FILENAMES_CONFIG['results'])
#         labeled_data, new_to_old_id_mapping = construct_labels(results)
#         io_module.write_config(FILENAMES_CONFIG['new_names'], new_to_old_id_mapping)
#
#         filenames, labels = labeled_data
#         numpy_images = []
#         for f in filenames:
#             # TODO: consider not reading images to memory and using 'np.memmap' instead
#             img = imread(f)
#             img = preprocess_image(img)
#             numpy_images.append(img)
#
#         return np.asarray(numpy_images), np.asarray(labels)
#
#     finally:
#         os.chdir(old_dir)
#
#
# def construct_labels(results):
#     """
#     Create appropriate image labels (a continuous range of integers starting from 0)
#     given results containing beer ids.
#     :return a pair containing list of image names and a corresponding list of labels
#             and a list 'new_mapping'. If new_mapping[new_label] = old_id, then beer
#             with id 'old_id' is mapped to a label 'new_label'.
#     """
#     filenames = []
#     labels = []
#     new_mapping = []
#     results.sort(key=itemgetter(1))
#
#     new_id = -1
#     last_old_id = -1
#     for res in results:
#         old_id = res[1]
#         if old_id != last_old_id:
#             last_old_id = old_id
#             new_id += 1
#             new_mapping.append(old_id)
#         filenames.append(res[0])
#         labels.append(new_id)
#
#     return (filenames, labels), new_mapping
#
#
# def preprocess_image(img):
#     """ Prepare an image to be a single training example. """
#     img = np.true_divide(img, 255.)
#     img = img.transpose([2, 0, 1])
#     return img
#
#
# def prepare_dataset(dataset, rng, train_ratio, validation_ratio):
#     """ Shuffle a dataset, turn to numpy arrays and split to train, validation and test sets. """
#     dataset = shuffle_lists(dataset[0], dataset[1], rng)
#     # turn lists to numpy arrays
#     dataset = np.asarray(dataset[0]), np.asarray(dataset[1])
#     # split data into training, validation and test sets
#     return split_data(dataset, train_ratio, validation_ratio)
#
#
# def shuffle_lists(list1, list2, rng):
#     """ Shuffle two corresponding lists. """
#     list1_shuf = []
#     list2_shuf = []
#     index_shuf = range(len(list1))
#     rng.shuffle(index_shuf)
#     for i in index_shuf:
#         list1_shuf.append(list1[i])
#         list2_shuf.append(list2[i])
#     return list1_shuf, list2_shuf
#
#
# def split_data(dataset, train_ratio, validation_ratio):
#     """ Split dataset to train, validation and test sets. """
#     data_size = len(dataset[0])
#     data_train_size = int(data_size * train_ratio)
#     data_validation_size = int(data_size * validation_ratio) + data_train_size
#
#     x, y = dataset
#     return [(x[:data_train_size], y[:data_train_size]),
#             (x[data_train_size:data_validation_size], y[data_train_size:data_validation_size]),
#             (x[data_validation_size:], y[data_validation_size:])]
#
#
# def shared_dataset(data_xy, borrow=True):
#     """ Load the dataset into shared variables. """
#     data_x, data_y = data_xy
#     shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
#     shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
#     return shared_x, T.cast(shared_y, 'int32')
#
#
# def print_statistics(datasets):
#     """ Print some statistics about datasets. """
#     for set_name, set in zip(['train', 'validation', 'test'], datasets):
#         print set_name, 'set: total count:', len(set[0])
#         print '    labels:', set[1]
