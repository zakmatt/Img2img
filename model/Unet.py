import tensorflow as tf

from .utils import (
    gen_conv,
    lrelu,
    batch_norm,
    gen_deconv,
    discrim_conv
)


class Unet(object):

    @staticmethod
    def generator(inputs, output_channels, n_filters):
        layers = []

        # encode [batch_size, 256, 256, input_chann] =>
        #        [batch_size, 128, 128, ngf]
        with tf.variable_scope('encode_1'):
            output = gen_conv(batch_input=inputs, out_channels=n_filters)
            layers.append(output)

        layers_specs = [
            n_filters * 2,
            n_filters * 4,
            n_filters * 8,
            n_filters * 8,
            n_filters * 8,
            n_filters * 8,
            n_filters * 8
        ]

        for output_channels in layers_specs:
            encoder_name = "encoder_{}".format(len(layers) + 1)
            with tf.variable_scope(encoder_name):
                rectified_inputs = lrelu(layers[-1], 0.2)
                convolved = gen_conv(rectified_inputs, output_channels)
                output = batch_norm(convolved)
                layers.append(output)

        layers_specs = [
            (n_filters * 8, 0.5),
            (n_filters * 8, 0.5),
            (n_filters * 8, 0.5),
            (n_filters * 8, 0.0),
            (n_filters * 4, 0.0),
            (n_filters * 2, 0.0),
            (n_filters, 0.0),
        ]

        num_encoder_layers = len(layers)
        for dec_layer, (output_channels, dropout) in enumerate(layers_specs):
            skip_layer = num_encoder_layers - dec_layer - 1
            decoder_name = "decoder_{}".format(dec_layer + 1)
            with tf.variable_scope(decoder_name):
                if dec_layer == 0:
                    # no skip connections for the first layer
                    inputs = layers[-1]
                else:
                    inputs = tf.concat(
                        [layers[-1], layers[skip_layer]],
                        axis=3
                    )

            rectified_inputs = tf.nn.relu(inputs)
            output = gen_deconv(rectified_inputs, output_channels)
            output = batch_norm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=(1 - dropout))

            layers.append(output)

        # decoder_1: [batch_size, 128, 128, n_filters * 2] =>
        #            [batch_size, 256, 256, output_channels]
        with tf.variable_scope('decoder_1'):
            inputs = tf.concat([layers[-1], layers[0]], axis=3)
            rectified_inputs = tf.nn.relu(inputs)
            output = gen_deconv(rectified_inputs, output_channels)
            output = tf.nn.tanh(output)
            layers.append(output)

        return layers[-1]


    @staticmethod
    def discriminator(inputs, targets, n_filters):
        n_layers = 3
        layers = []
        # 2x [batch, height, width, in_channels] =>
        #    [batch, height, width, in_channels * 2]
        inputs = tf.concat([inputs, targets], axis=3)

        with tf.variable_scope('layer_1'):
            convolved = discrim_conv(inputs, n_filters, stride=2)
            rectified_output = lrelu(convolved, 0.2)
            layers.append(rectified_output)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            layer_name = 'layer_{}'.format(len(layers) + 1)
            with tf.variable_scope(layer_name):
                output_channels = n_filters * min(2**(i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2
                convolved = discrim_conv(layers[-1], output_channels, stride)
                normalized = batch_norm(convolved)
                rectified_output = lrelu(normalized, 0.2)
                layers.append(rectified_output)

        layer_name = 'layer_{}'.format(len(layers) + 1)
        with tf.variable_scope(layer_name):
            convolved = discrim_conv(layers[-1], out_channels=1, stride=1)
            output = tf.nn.sigmoid(convolved)
            layers.append(output)



    @staticmethod
    def model(inputs, targets):
        pass
