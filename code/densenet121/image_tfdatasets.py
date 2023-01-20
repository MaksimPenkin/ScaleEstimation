import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from imgtf_utils import read_img_tf
from image_samplers import ImagePathLabelSampler, ArraysSampler
from constants import imagenet_mean, imagenet_std, cifar10_mean, cifar10_std


class PATH_DT_MSU_WSI:

    def __init__(self, db_list, db_folder_to_label_list, db_basedir="", dtype_in=tf.uint8, shuffle=False):

        path_sampler = ImagePathLabelSampler(db_list, db_folder_to_label_list, basedir=db_basedir, shuffle=shuffle)

        self.gen = lambda: (x for x in path_sampler)
        self.dtype_in = dtype_in
        self.num_samples = len(path_sampler)

    def __len__(self):
        return self.num_samples

    def get_tfdataset(self, with_label=True, batch_size=32):
        if with_label:
            tf_dataset = (tf.data.Dataset.from_generator(self.gen,
                                                         output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                           tf.TensorSpec(shape=(1,), dtype=tf.int32)))
                          .map(lambda x, y: (read_img_tf(read_path=x,
                                                         dtype=self.dtype_in,
                                                         name="input_img",
                                                         expand_dims=True), y))  # read image and explicit batch-dim add
                          .map(lambda x, y: (tf.cast(x[0], tf.float32), tf.cast(y[0], tf.int32)))  # return Image, Label
                          .batch(batch_size=batch_size)
                          .prefetch(1)
                          )
        else:
            assert batch_size == 1
            tf_dataset = (tf.data.Dataset.from_generator(self.gen,
                                                         output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                           tf.TensorSpec(shape=(1,), dtype=tf.int32)))
                          .map(lambda x, y: (read_img_tf(read_path=x,
                                                         dtype=self.dtype_in,
                                                         name="input_img",
                                                         expand_dims=True), y))  # read image and explicit batch-dim add
                          .map(lambda x: tf.cast(x, tf.float32))  # return Image
                          )

        return tf_dataset


class ImageNet1k:

    def __init__(self, db_list, db_folder_to_label_list, db_basedir="", dtype_in=tf.uint8, shuffle=False):

        path_sampler = ImagePathLabelSampler(db_list, db_folder_to_label_list, basedir=db_basedir, shuffle=shuffle)

        self.gen = lambda: (x for x in path_sampler)
        self.dtype_in = dtype_in
        self.num_samples = len(path_sampler)

        self.mean = tf.constant(imagenet_mean[np.newaxis, np.newaxis, np.newaxis, ...],
                                dtype=tf.float32,
                                name="imagenet_mean")

        self.std = tf.constant(imagenet_std[np.newaxis, np.newaxis, np.newaxis, ...],
                               dtype=tf.float32,
                               name="imagenet_std")

    def __len__(self):
        return self.num_samples

    def get_tfdataset(self, with_label=True, batch_size=32):
        if with_label:
            tf_dataset = (tf.data.Dataset.from_generator(self.gen,
                                                         output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                           tf.TensorSpec(shape=(1,), dtype=tf.int32)))
                          .map(lambda x, y: (read_img_tf(read_path=x,
                                                         dtype=self.dtype_in,
                                                         name="input_img",
                                                         expand_dims=True), y))  # read image and explicit batch-dim add
                          .map(lambda x, y: ((x - self.mean) / self.std, y))  # normalization
                          .map(lambda x, y: (tf.cast(x[0], tf.float32), tf.cast(y[0], tf.int32)))  # return Image, Label
                          .batch(batch_size=batch_size)
                          .prefetch(1)
                          )
        else:
            assert batch_size == 1
            tf_dataset = (tf.data.Dataset.from_generator(self.gen,
                                                         output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                           tf.TensorSpec(shape=(1,), dtype=tf.int32)))
                          .map(lambda x, y: (read_img_tf(read_path=x,
                                                         dtype=self.dtype_in,
                                                         name="input_img",
                                                         expand_dims=True), y))  # read image and explicit batch-dim add
                          .map(lambda x, y: (x - self.mean) / self.std)  # normalization
                          .map(lambda x: tf.cast(x, tf.float32))  # return Image
                          )

        return tf_dataset


class CIFAR10:

    def __init__(self, db_list=None, shuffle=False):

        if db_list is None:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        else:
            x_train = np.load(os.path.join(db_list, "x_train.npy"))
            y_train = np.load(os.path.join(db_list, "y_train.npy"))

            x_test = np.load(os.path.join(db_list, "x_test.npy"))
            y_test = np.load(os.path.join(db_list, "y_test.npy"))

        assert x_train.shape == (50000, 32, 32, 3) and x_train.dtype == np.uint8
        assert x_test.shape == (10000, 32, 32, 3) and x_test.dtype == np.uint8
        assert y_train.shape == (50000, 1)
        assert y_test.shape == (10000, 1)

        train_sampler = ArraysSampler([x_train, y_train], shuffle=shuffle)
        test_sampler = ArraysSampler([x_test, y_test], shuffle=False)

        self.gen_train = lambda: (x for x in train_sampler)
        self.gen_test = lambda: (x for x in test_sampler)

        self.num_samples_train = len(train_sampler)
        self.num_samples_test = len(test_sampler)

        self.mean = tf.constant(cifar10_mean[np.newaxis, np.newaxis, np.newaxis, ...],
                                dtype=tf.float32,
                                name="cifar10_mean")

        self.std = tf.constant(cifar10_std[np.newaxis, np.newaxis, np.newaxis, ...],
                               dtype=tf.float32,
                               name="cifar10_std")

    @property
    def num_samples(self):
        return {"train": self.num_samples_train,
                "test": self.num_samples_test}

    def get_tfdataset(self, db="train", with_label=True, batch_size=32):
        gen = self.gen_train if db == "train" else self.gen_test

        if with_label:
            tf_dataset = (tf.data.Dataset.from_generator(gen,
                                                         output_signature=(tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8),
                                                                           tf.TensorSpec(shape=(1,), dtype=tf.uint8)))
                          .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.int32)))
                          .map(lambda x, y: ((x - self.mean) / self.std, y))  # normalization and implicit batch-dim add, by broadcasting
                          .map(lambda x, y: (tf.cast(x[0], tf.float32), tf.cast(y[0], tf.int32)))  # return Image, Label
                          .batch(batch_size=batch_size)
                          .prefetch(1)
                          )
        else:
            assert batch_size == 1
            tf_dataset = (tf.data.Dataset.from_generator(gen,
                                                         output_signature=(tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8),
                                                                           tf.TensorSpec(shape=(1,), dtype=tf.uint8)))
                          .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.int32)))
                          .map(lambda x, y: (x - self.mean) / self.std)  # normalization and implicit batch-dim add, by broadcasting
                          .map(lambda x: tf.cast(x, tf.float32))  # return Image
                          )

        return tf_dataset
