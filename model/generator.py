from layers.convolution_layer import ConvolutionLayer
from keras import initializers
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# train_X, tmp_x, ngf, filter_size, image_width, image_height, input_channel, output_channel, batch_size
class Generator(object):
    def __init__(self, x_train, params):
        self.x_train = x_train
        self.params = params

    def create_model(self):
        input_channels = self.params['input_channels']
        img_width = self.params['img_width']
        img_height = self.params['img_height']

        enc_dec_layers = []

        layer_params = {
            'ngf': self.params['ngf'],
            'filter_size': (self.params['filter_size'], self.params['filter_size']),
            'strides': (2, 2),
            'padding': 'same',
            'kernel_initializer': 'random_normal',
            'input_shape': (64, img_width, img_height, input_channels)
        }

        # (256, 256) => (128, 128)

        l_0 = ConvolutionLayer(self.x_train, layer_params)
        leaky_relu = LeakyReLU(alpha=0.2)
        encoder_conv_output_0, encoder_processed_output_0 = l_0.process(activation=leaky_relu)

        # (128, 128) => (64, 64)
        layer_params['ngf'] *= 2
        l_1 = ConvolutionLayer(encoder_processed_output_0, layer_params)
        enc_bn_1 = BatchNormalization(epsilon=1e-5, momentum=0.9)
        encoder_conv_output_1, encoder_processed_output_1 = l_1.process(activation=leaky_relu,
                                                                        batch_norm=enc_bn_1)

        # (64, 64) => (32, 32)
        layer_params['ngf'] *= 2
        l_2 = ConvolutionLayer(encoder_processed_output_1, layer_params)
        enc_bn_2 = BatchNormalization(epsilon=1e-5, momentum=0.9)
        encoder_conv_output_2, encoder_processed_output_2 = l_2.process(activation=leaky_relu,
                                                                        batch_norm=enc_bn_2)

        # (32, 32) => (16, 16)
        layer_params['ngf'] *= 2
        l_3 = ConvolutionLayer(encoder_processed_output_2, layer_params)
        enc_bn_3 = BatchNormalization(epsilon=1e-5, momentum=0.9)
        encoder_conv_output_3, encoder_processed_output_3 = l_3.process(activation=leaky_relu,
                                                                        batch_norm=enc_bn_3)

        # (16, 16) => (8, 8)
        layer_params['ngf'] *= 2
        l_4 = ConvolutionLayer(encoder_processed_output_3, layer_params)
        enc_bn_4 = BatchNormalization(epsilon=1e-5, momentum=0.9)
        encoder_conv_output_4, encoder_processed_output_4 = l_4.process(activation=leaky_relu,
                                                                        batch_norm=enc_bn_4)


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf

    input_array = np.random.randn(64, 256, 256, 3)
    img_shape = list(input_array[0].shape)
    training_img_data = tf.placeholder(tf.float32, shape=[None] + img_shape)
    training_label_data = tf.placeholder(tf.float32, shape=[2] + img_shape)
    queue = tf.FIFOQueue(shapes=[img_shape, img_shape],
                         dtypes=[tf.float32, tf.float32],
                         capacity=2000)
    enqueue_ops = queue.enqueue_many([training_label_data, training_img_data])
    labels, imgs = queue.dequeue_many(64)
    layer_params = {
        'ngf': 64,
        'filter_size': 4,
        'input_channels': img_shape[2],
        'img_width': img_shape[0],
        'img_height': img_shape[1]
    }
    gen = Generator(labels, layer_params)
    gen.create_model()
