import tensorflow as tf


def gen_conv(batch_input, out_channels):
    # [batch_size, in_width, in_height, in_channels] =>
    # [batch_size, out_width, out_height, out_channels]

    # tensor with a normal distribution
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(
        batch_input,
        out_channels,
        kernel_size=2,
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )


def gen_deconv(batch_input, out_channels):
    # [batch_size, in_width, in_height, in_channels] =>
    # [batch_size, in_width, in_height, out_channels]

    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(
        batch_input,
        out_channels,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(
        batch_input,
        [[0, 0], [1, 1], [1, 1], [0, 0]],
        mode='CONSTANT'
    )
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(
        padded_input,
        out_channels,
        kernel_size=4,
        strides=(stride, stride),
        padding='valid',
        kernel_initializer=initializer
    )


def lrelu(x, a):
    with tf.variable_scope('relu'):
        return tf.nn.leaky_relu(
            features=x,
            alpha=a
        )


def batch_norm(inputs):
    initializer = tf.random_normal_initializer(1.0, 0.02)
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        epsilon=1e-5,
        momentum=0.1,
        training=True,
        gamma_initializer=initializer
    )