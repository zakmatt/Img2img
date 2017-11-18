import h5py
import numpy as np
import os


class Data:
    def __init__(self, data_path, set_preffix):
        self.images_path = os.path.join(data_path, "target_{0}_dataset.npy".format(set_preffix))
        self.labels_path = os.path.join(data_path, "label_{0}_dataset.npy".format(set_preffix))
        self.__read_data()

    def __read_data(self):
        normalize = lambda x: (x / 127.5) - 1
        self.targets = normalize(np.load(self.images_path))
        self.labels = normalize(np.load(self.labels_path))
        self.__img_size = self.targets[0].shape

    def get_image_size(self):
        return self.__img_size

    def get_data(self):
        return self.labels.astype(np.float32), self.targets.astype(np.float32)

    def data_amount(self):
        return len(self.labels)