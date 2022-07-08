import tensorflow as tf
import numpy as np
import os, json, time, threading, shutil, zipfile
from sklearn.model_selection import train_test_split
import numpy.lib.format as fmt
from tempfile import mkdtemp
from collections import Counter
import math
from tensorflow.keras.callbacks import Callback

# Loads a features file, mmap's if size is above 128MiB
def np_load_file_auto_mmap(file):
    # 128MiB seems like a safe number to always load fully into memory
    # e.g. small feature sets, labels, etc. we can probably up it if we actually calculate the used memory
    if (os.path.getsize(file) > 128 * 1024 * 1024):
        return np.load(file, mmap_mode='r')
    else:
        return np.load(file)

# Since the data files are too large to load into memory, we must split and shuffle them by writing their contents to
# new files in random order. The new files can be memory mapped in turn.
def split_and_shuffle_data(y_type, classes, classes_values, mode, seed, dir_path, test_size=0.2,
                           X_train_features_path='X_train_features.npy', y_train_path='y_train.npy',
                           X_train_raw_path=None, stratify_sample=False):

    # This is where the split data will be written
    X_train_output_path = os.path.join(os.sep, 'tmp', 'X_split_train.npy')
    X_train_raw_output_path = os.path.join(os.sep, 'tmp', 'X_split_train_raw.npy')
    X_test_output_path = os.path.join(os.sep, 'tmp', 'X_split_test.npy')
    Y_train_output_path = os.path.join(os.sep, 'tmp', 'Y_split_train.npy')
    Y_test_output_path = os.path.join(os.sep, 'tmp', 'Y_split_test.npy')

    X = None
    X_raw = None
    Y = None

    # assume we are not stratifying labels by default.
    split_stratify_labels = None

    # Load the training data
    if y_type == 'structured':
        if X_train_raw_path:
            raise Exception('Raw input is not yet supported for structured Y data')

        X = np_load_file_auto_mmap(os.path.join(dir_path, 'X_train_features.npy'))
        Y = ei_tensorflow.utils.load_y_structured(dir_path, 'y_train.npy', len(X))

    elif y_type == 'npy':
        X = np_load_file_auto_mmap(os.path.join(dir_path, X_train_features_path))
        Y = np_load_file_auto_mmap(os.path.join(dir_path, y_train_path))[:,0]
        if X_train_raw_path:
            X_raw = np_load_file_auto_mmap(os.path.join(dir_path, X_train_raw_path))

        # If we are building the sample with stratification; i.e, strictly enforcing that
        # the label distribution matches between train/test; we need to record the raw Y
        # values now before they are mutated.
        if stratify_sample:
            split_stratify_labels = Y

        # Do this before writing new splits to disk so that the resulting mmapped array has categorical values,
        # otherwise it will be difficult to generate performance stats, which depend on numpy arrays.
        if mode == 'classification':
            Y = tf.keras.utils.to_categorical(Y - 1, num_classes=classes)
        elif mode == 'regression':
            # for regression we want to map to the real values
            Y = np.array([ float(classes_values[y - 1]) for y in Y ])
            if np.isnan(Y).any():
                print('Your dataset contains non-numeric labels. Cannot train regression model.')
                exit(1)

    X_ids = list(range(len(X)))
    Y_ids = list(range(len(Y)))
    X_train_ids, X_test_ids, Y_train_ids, Y_test_ids = train_test_split(X_ids, Y_ids, test_size=test_size,
                                                                        random_state=seed,
                                                                        stratify=split_stratify_labels)

    # Generates a header for the .npy file
    def get_header(array, new_length):
        new_shape = (new_length,) + array.shape[1:]
        return {'descr': fmt.dtype_to_descr(array.dtype), 'fortran_order': False, 'shape': new_shape}

    # Saves a subset of an array's indexes to a numpy file, the subset and order specified by an array of ints
    def save_to_npy(array, indexes, file_path):
        header = get_header(array, len(indexes))
        with open(file_path, 'wb') as f:
            fmt.write_array_header_2_0(f, header)
            for ix in indexes:
                f.write(array[ix].tobytes('C'))

    save_to_npy(X, X_train_ids, X_train_output_path)
    save_to_npy(X, X_test_ids, X_test_output_path)

    if X_train_raw_path:
        # We only need the train split for the raw data, since test will not have augmentations applied
        save_to_npy(X_raw, X_train_ids, X_train_raw_output_path)
        X_train_raw = np_load_file_auto_mmap(X_train_raw_output_path)
    else:
        X_train_raw = None

    X_train = np_load_file_auto_mmap(X_train_output_path)
    X_test = np_load_file_auto_mmap(X_test_output_path)

    if  y_type == 'structured':
        # The structured data is just handled in memory.
        # Load these from JSON and then split Y_structured_train using the same method as above.
        Y_train = [Y[i] for i in Y_train_ids]
        Y_test = [Y[i] for i in Y_test_ids]
    elif y_type == 'npy':
        save_to_npy(Y, Y_train_ids, Y_train_output_path)
        save_to_npy(Y, Y_test_ids, Y_test_output_path)
        Y_train = np_load_file_auto_mmap(Y_train_output_path)
        Y_test = np_load_file_auto_mmap(Y_test_output_path)

    return X_train, X_test, Y_train, Y_test, X_train_raw

# Feeds values from our memory mapped training data into the tensorflow dataset
def create_generator_standard(X_values, Y_values):
    data_length = len(X_values)
    def gen():
        for ix in range(data_length):
            yield X_values[ix], Y_values[ix]
    return gen

def get_dataset_standard(X_values, Y_values):
    memory_used = X_values.size * X_values.itemsize

    # if we have <1GB of data then just load into memory
    # the train jobs have at least 8GiB of RAM, so this should be fine
    if (memory_used < 1 * 1024 * 1024 * 1024):
        return tf.data.Dataset.from_tensor_slices((X_values, Y_values))
    else:
        # Using the 'args' param of 'from_generator' results in a memory leak, so we instead use a function that
        # returns a generator that wraps the data arrays.
        return tf.data.Dataset.from_generator(create_generator_standard(X_values, Y_values),
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=(tf.TensorShape(X_values[0].shape),
                                                            tf.TensorShape(Y_values[0].shape)))

def get_reshape_function(reshape_to):
    def reshape(image, label):
        return tf.reshape(image, reshape_to), label
    return reshape
