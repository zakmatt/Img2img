from keras.layers import Activation, ZeroPadding2D
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

        padding = (1, 1)
        patchGAN.add(ZeroPadding2D(
            padding=padding
        ))
        # C512 32=>16
        patchGAN.add(Conv2D(ndf * 8,
                                   kernel_size=self.params['kernel_size'],
                                   kernel_initializer='random_normal',
                                   ))
        patchGAN.add(BatchNormalization())
        patchGAN.add(LeakyReLU(alpha=0.2))

        patchGAN.add(ZeroPadding2D(
            padding=padding
        ))

        patchGAN.add(Conv2D(1,
                                   kernel_size=self.params['kernel_size'],
                                   kernel_initializer='random_normal',
                                   ))
        patchGAN.add(Activation('sigmoid'))
        return patchGAN

if __name__=='__main__':
    import numpy as np
    import tensorflow as tf
    from keras.layers import Concatenate
    from keras import optimizers
    from keras import backend as K

    labels = np.random.randn(64, 256, 256, 3)
    imgs = np.random.randn(64, 256, 256, 3)
    images_shape = labels[0].shape
    '''
    training_img_data = tf.placeholder(tf.float32, shape=[None] + img_shape)
    training_label_data = tf.placeholder(tf.float32, shape=[None] + img_shape)
    queue = tf.FIFOQueue(shapes=[img_shape, img_shape],
                         dtypes=[tf.float32, tf.float32],
                         capacity=2000)
    enqueue_ops = queue.enqueue_many([training_label_data, training_img_data])
    
    labels, imgs = queue.dequeue_many(64)
    '''
    input = np.concatenate((labels, imgs), axis=-1)
    layer_params = {
        'ndf': 64,
        'kernel_size': (4, 4),
        'input_shape': (images_shape[0], images_shape[1], images_shape[2] * 2)
    }
    disc = Discriminator(input, layer_params)
    discriminator = disc.create_model()
    output = discriminator.predict(input)
    print(output.shape)


    def discriminator_on_generator_loss(y_true, y_pred):
        cross_entropy = K.binary_crossentropy(y_pred, y_true)
        return K.mean(cross_entropy, axis=-1)

    #adam_optimizer = optimizers.Adam(lr=0.0002, beta_1=0.5)
    discriminator.trainable = True
    discriminator.compile(optimizer='RMSprop', loss=discriminator_on_generator_loss)

