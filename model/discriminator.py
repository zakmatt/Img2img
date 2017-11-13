from keras.layers import Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from model import Model


class Discriminator(Model):
    def create_model(self, batch_size=64):
        ndf = self.params['ndf']
        patchGAN = Sequential()

        # C64 256=>128
        patchGAN.add(Conv2D(filters=ndf,
                            kernel_size=self.params['kernel_size'],
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer='random_normal',
                            input_shape=self.params['input_shape'],
                            ))
        patchGAN.add(LeakyReLU(alpha=0.2))

        # C128 128=>64
        patchGAN.add(Conv2D(ndf * 2,
                                   kernel_size=self.params['kernel_size'],
                                   strides=(2, 2),
                                   padding='same',
                                   kernel_initializer='random_normal',
                                   ))
        patchGAN.add(BatchNormalization())
        patchGAN.add(LeakyReLU(alpha=0.2))

        # C256 64=>32
        patchGAN.add(Conv2D(ndf * 4,
                                   kernel_size=self.params['kernel_size'],
                                   strides=(2, 2),
                                   padding='same',
                                   kernel_initializer='random_normal',
                                   ))
        patchGAN.add(BatchNormalization())
        patchGAN.add(LeakyReLU(alpha=0.2))

        # C512 32=>16
        patchGAN.add(Conv2D(ndf * 8,
                                   kernel_size=self.params['kernel_size'],
                                   strides=(1, 1),
                                   padding='same',
                                   kernel_initializer='random_normal',
                                   ))
        patchGAN.add(BatchNormalization())
        patchGAN.add(LeakyReLU(alpha=0.2))

        patchGAN.add(Conv2D(1,
                                   kernel_size=self.params['kernel_size'],
                                   strides=(1, 1),
                                   padding='same',
                                   kernel_initializer='random_normal',
                                   ))
        patchGAN.add(Activation('sigmoid'))
        patchGAN.add(Flatten())
        return patchGAN
        '''
        ndf = self.params['ndf']

        ndf_values = [
            2 * ndf,
            4 * ndf,
            8 * ndf
        ]

        model = Sequential()

        model.add(
            Conv2D(
                filters=ndf,
                kernel_size=self.params['kernel_size'],
                strides=(2, 2),
                padding='same',
                kernel_initializer='random_normal',
                input_shape=self.params['input_shape']
            )
        )
        model.add(LeakyReLU(alpha=0.2))

        for pos, ndf in enumerate(ndf_values):
            model.add(Conv2D(
                filters=ndf,
                kernel_size=self.params['kernel_size'],
                strides=(2, 2) if pos != len(ndf_values)-1 else (1, 1),
                padding='same',
                kernel_initializer='random_normal'
            ))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(
            filters=1,
            kernel_size=self.params['kernel_size'],
            strides=(1, 1),
            padding='same',
            kernel_initializer='random_normal'
        ))
        model.add(Activation('sigmoid'))
        model.add(Flatten())

        self.model = model
        '''

if __name__=='__main__':
    import numpy as np
    import tensorflow as tf
    input_array = np.random.randn(64, 256, 256, 3)
    img_shape = list(input_array[0].shape)
    training_img_data = tf.placeholder(tf.float32, shape=[None] + img_shape)
    training_label_data = tf.placeholder(tf.float32, shape=[None] + img_shape)
    queue = tf.FIFOQueue(shapes=[img_shape, img_shape],
                         dtypes=[tf.float32, tf.float32],
                         capacity=2000)
    enqueue_ops = queue.enqueue_many([training_label_data, training_img_data])
    labels, imgs = queue.dequeue_many(64)
    input = tf.concat([labels, imgs], axis=3)
    layer_params = {
        'ndf': 64,
        'kernel_size': (4, 4),
        'input_shape': (img_shape[0], img_shape[1], img_shape[2] * 2)
    }
    disc = Discriminator(input, layer_params)
    disc.create_model()
    output = disc.model(input)
    print(output.shape)


