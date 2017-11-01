import h5py
import numpy as np


class Data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.__read_data()

    def __read_data(self):
        normalize = lambda x: (x / 127.5) - 1
        with h5py.File(self.data_path, "r") as file:
            X_train = file["train_data_full"][:].astype(np.float32)
            X_train = normalize(X_train)

            X_sketch_train = file["train_data_sketch"][:].astype(np.float32)
            X_sketch_train = normalize(X_sketch_train)

            # transpose to shape (# of samples, width, height, # of channels)
            self.X_train = X_train.transpose(0, 2, 3, 1)
            self.X_sketch_train = X_sketch_train.transpose(0, 2, 3, 1)

            X_val = file["val_data_full"][:].astype(np.float32)
            X_val = normalize(X_val)

            X_sketch_val = file["val_data_sketch"][:].astype(np.float32)
            X_sketch_val = normalize(X_sketch_val)

            self.X_val = X_val.transpose(0, 2, 3, 1)
            self.X_sketch_val = X_sketch_val.transpose(0, 2, 3, 1)

            self.__img_size = list(self.X_train[0].shape)


    def get_image_size(self):
        return self.__img_size