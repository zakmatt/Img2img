import tensorflow as tf

from collections import namedtuple

from utils.model_utils import (
    gen_conv,
    lrelu,
    batch_norm,
    gen_deconv,
    discrim_conv
)

EPS = 1e-12
Model = namedtuple(
    'Model',
    ['outputs',
     'predict_real',
     'predict_fake',
     'discrim_loss',
     'discrim_grads_and_vars',
     'gen_loss_GAN',
     'gen_loss_L1',
     'gen_grads_and_vars',
     'train']
)


class Unet(object):
    @staticmethod
    def generator(inputs, out_channels, n_filters):
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
        #            [batch_size, 256, 256, out_channels]
        with tf.variable_scope('decoder_1'):
            inputs = tf.concat([layers[-1], layers[0]], axis=3)
            rectified_inputs = tf.nn.relu(inputs)
            output = gen_deconv(rectified_inputs, out_channels)
            output = tf.nn.tanh(output)
            layers.append(output)

        return layers[-1]

    @staticmethod
    def generator_loss(predict_fake,
                       targets,
                       generated,
                       gan_weight=1.0,
                       l1_weight=100):
        with tf.variable_scope('generator_loss'):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_gan = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_l1 = tf.reduce_mean(tf.abs(targets - generated))
            gen_loss = gen_loss_gan * gan_weight + gen_loss_l1 * l1_weight
            return gen_loss, gen_loss_l1, gen_loss_gan

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
                output_channels = n_filters * min(2 ** (i + 1), 8)
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

        return layers[-1]

    @staticmethod
    def discriminator_loss(predict_real, predict_fake):
        with tf.variable_scope('discriminator_loss'):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            return tf.reduce_mean(
                -(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS))
            )

    @staticmethod
    def model(inputs, targets):
        """

        Create a U-Net model with skip connections

        :param inputs: Input images
        :type inputs: tensorflow.placeholder
        :param targets: targeting images
        :type targets: tensorflow.placeholder
        :return:
        """
        with tf.variable_scope('generator'):
            output_channels = int(targets.get_shape()[-1])
            generated = Unet.generator(inputs, output_channels, 64)

        # create two copies of discriminator, one for real pairs
        # and one for fake pairs. They share the same underlying variables
        with tf.variable_scope('discriminator'):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = Unet.discriminator(inputs, targets, 64)

        with tf.variable_scope('discriminator', reuse=True):
            predict_fake = Unet.discriminator(inputs, generated, 64)

        discriminator_loss = Unet.discriminator_loss(
            predict_real,
            predict_fake
        )
        generator_loss, generator_loss_l1, gen_loss_gan = Unet.generator_loss(
            predict_fake,
            targets,
            generated
        )

        with tf.variable_scope('discriminator_train'):
            discrim_train_vars = [
                var for var in tf.trainable_variables()
                if var.name.startswith('discriminator')
            ]
            discrim_optim = tf.train.AdamOptimizer(0.0002, 0.5)
            discrim_grads_vars = discrim_optim.compute_gradients(
                discriminator_loss,
                var_list=discrim_train_vars
            )
            discrim_train = discrim_optim.apply_gradients(discrim_grads_vars)

        with tf.variable_scope('generator_train'):
            with tf.control_dependencies([discrim_train]):
                gen_train_vars = [
                    var for var in tf.trainable_variables()
                    if var.name.startswith('generator')
                ]
                gen_optim = tf.train.AdamOptimizer(0.0002, 0.5)
                gen_grads_vards = gen_optim.compute_gradients(
                    generator_loss,
                    var_list=gen_train_vars
                )
                gen_train = gen_optim.apply_gradients(gen_grads_vards)

        exp_moving_average = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = exp_moving_average.apply(
            [discriminator_loss, gen_loss_gan, generator_loss_l1]
        )

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=exp_moving_average.average(discriminator_loss),
            discrim_grads_and_vars=discrim_grads_vars,
            gen_loss_GAN=exp_moving_average.average(gen_loss_gan),
            gen_loss_L1=exp_moving_average.average(generator_loss_l1),
            gen_grads_and_vars=gen_grads_vards,
            outputs=generated,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )
