import os
import json
import random
import numpy as np


class ImagePathSampler:

    def __init__(self, image_list,
                 basedir="",
                 shuffle=True):

        self.basedir = basedir
        with open(image_list, "rt") as f:
            self.image_list = f.read().splitlines()
        self.image_list = list(map(lambda x: [os.path.join(self.basedir, y) for y in x.split(" ")],
                                   self.image_list))

        self.num_samples = len(self.image_list)

        self.shuffle = shuffle
        self._reset()

    def _reset(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.image_list)

    def __iter__(self):
        self._reset()
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.pos >= self.num_samples:
            self._reset()

        one_sample = self.image_list[self.pos]
        self.pos = self.pos + 1

        if len(one_sample) == 1:
            return one_sample[0]
        else:
            return tuple(one_sample)

    def __call__(self):
        return self.__iter__()


class ImagePathLabelSampler:

    def __init__(self, image_list,
                 folder_to_label_list,
                 basedir="",
                 shuffle=True):

        self.basedir = basedir
        with open(image_list, "rt") as f:
            self.image_list = f.read().splitlines()
        self.image_list = [os.path.join(self.basedir, x) for x in self.image_list]

        with open(folder_to_label_list, "r") as f:
            self.folder_to_label = json.load(f)

        self.num_samples = len(self.image_list)

        self.shuffle = shuffle
        self._reset()

    def _reset(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.image_list)

    def __iter__(self):
        self._reset()
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.pos >= self.num_samples:
            self._reset()

        p1 = self.image_list[self.pos]
        p2 = os.path.split(os.path.split(p1)[0])[-1]  # take folder name, as for class name
        p2 = int(self.folder_to_label[p2])  # convert class name into integer label

        self.pos = self.pos + 1
        return p1, (p2, )

    def __call__(self):
        return self.__iter__()


class ArraysSampler:

    def __init__(self, arr_list,
                 shuffle=True):

        self.arr_list = arr_list
        self.num_samples = min([len(x) for x in self.arr_list])
        self.indices = np.arange(self.num_samples)

        self.shuffle = shuffle
        self._reset()

    def _reset(self):
        self.pos = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self._reset()
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.pos >= self.num_samples:
            self._reset()

        one_sample = []
        for i in range(len(self.arr_list)):
            idx = self.indices[self.pos]
            x = self.arr_list[i][idx]
            one_sample.append(x)
        self.pos = self.pos + 1

        if len(one_sample) == 1:
            return one_sample[0]
        else:
            return tuple(one_sample)

    def __call__(self):
        return self.__iter__()
