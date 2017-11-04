from keras.layers import Activation, Dropout, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from layers.convolution_layer import ConvolutionLayer
from layers.deconvolution_layer import DeconvolutionLayer

from model import Model
import tensorflow as tf

class Generator(Model):

    def create_model(self, batch_size=64):
        input_channels = self.params['input_channels']
        img_width = self.params['img_width']
        img_height = self.params['img_height']

        layer_params = {
            'ngf': self.params['ngf'],
            'filter_size': (self.params['filter_size'], self.params['filter_size']),
            'strides': (2, 2),
            'padding': 'same',
            'kernel_initializer': 'random_normal',
            'input_shape': (batch_size, img_width, img_height, input_channels)
        }


        #change it to arrays!
        ngf_values = [
            self.params['ngf'] * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.params['ngf'] * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.params['ngf'] * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.params['ngf'] * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.params['ngf'] * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.params['ngf'] * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.params['ngf'] * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
        layers = []
        encoder_layers = []
        # (256, 256) => (128, 128)
        l_0 = ConvolutionLayer(self.x_train, layer_params)
        leaky_relu = LeakyReLU(alpha=0.2)

        encoder_conv_output, encoder_processed_output_0 = l_0.process(activation=leaky_relu)
        encoder_layers.append(encoder_conv_output)
        layers.append(encoder_processed_output_0)

        for ngf in ngf_values:

            layer_params['input_shape'] = (
                batch_size,
                layer_params['input_shape'][1]//2,
                layer_params['input_shape'][2]//2,
                layer_params['ngf']
            )
            layer_params['ngf'] = ngf
            layer = ConvolutionLayer(layers[-1], layer_params)
            batch_norm = BatchNormalization(epsilon=1e-5, momentum=0.9)
            convolved, processed = layer.process(activation=leaky_relu, batch_norm=batch_norm)
            encoder_layers.append(convolved)
            layers.append(processed)



        layer_specs = [
            (self.params['ngf'] * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (self.params['ngf'] * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (self.params['ngf'] * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (self.params['ngf'] * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (self.params['ngf'] * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (self.params['ngf'] * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (self.params['ngf'], 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        relu = Activation('relu')
        for decode_pos, (ngf, dropout) in enumerate(layer_specs):
            if decode_pos == 0:
                input = encoder_layers[-1]
            else:
                input = Concatenate(-1)([layers[-1], encoder_layers[-decode_pos-1]])

            input = relu(input)

            layer_params['input_shape'] = (
                batch_size,
                layer_params['input_shape'][1] if decode_pos == 0 else layer_params['input_shape'][1] * 2,
                layer_params['input_shape'][2] if decode_pos == 0 else layer_params['input_shape'][2] * 2,
                layer_params['ngf'] if decode_pos == 0 else layer_params['ngf'] * 2
            )
            layer_params['ngf'] = ngf
            layer = DeconvolutionLayer(input, layer_params)
            batch_norm = BatchNormalization(epsilon=1e-5, momentum=0.9)

            if dropout > 0.0:
                convolved = layer.process(batch_norm=batch_norm, dropout=Dropout(dropout))
            else:
                convolved = layer.process(batch_norm=batch_norm)

            layers.append(convolved)

        input = tf.concat([layers[-1], encoder_layers[0]], axis=3)
        input = relu(input)

        layer_params['input_shape'] = (
            batch_size,
            layer_params['input_shape'][1] * 2,
            layer_params['input_shape'][2] * 2,
            layer_params['ngf'] * 2
        )
        layer_params['ngf'] = input_channels
        layer = DeconvolutionLayer(input, layer_params)
        _, processed = layer.process(activation=Activation('tanh'))

        return processed




if __name__ == '__main__':
    import numpy as np
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
    output = gen.create_model()
    print(output.shape)
