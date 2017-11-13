from data import Data
import numpy as np
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator


def generate_batch():
    pass


def create_discriminator(image_size):
    layer_params = {
        'ndf': 64,
        'kernel_size': (4, 4),
        'input_shape': (image_size[0], image_size[1], image_size[2] * 2)
    }
    discriminator = Discriminator(None, layer_params)
    return discriminator.create_model()
    #return discriminator.model


def generator_image(labels, image_size, batch_size=64):
    layer_params = {
        'ngf': 64,
        'filter_size': 4,
        'input_channels': image_size[2],
        'img_width': image_size[0],
        'img_height': image_size[1]
    }

    generator = Generator(labels, layer_params)
    generated_img, params = generator.create_model(batch_size=batch_size)
    return generated_img, params


if __name__ == '__main__':
    batch_size = 64
    train_data = Data("../data/facades/", "train")
    image_size = train_data.get_image_size()
    train_labels, train_targets = train_data.get_data()

    val_data = Data("../data/facades/", "val")
    val_labels, val_targets = val_data.get_data()
    print(train_labels.shape)

    '''
    import cv2
    super_file = ((train_labels[0]+1)*127.5).astype(np.uint8)
    r, g, b = cv2.split(super_file)
    super_file = cv2.merge((b, g, r))
    cv2.imwrite('super_file.jpg', super_file)
    '''
    # Create placeholders
    training_labels_data = tf.placeholder(tf.float32, shape=[None] + list(image_size))
    training_img_data = tf.placeholder(tf.float32, shape=[None] + list(image_size))

    # Create queue
    queue = tf.FIFOQueue(shapes=[list(image_size), list(image_size)],
                         dtypes=[tf.float32, tf.float32],
                         capacity=2000)
    enqueue_ops = queue.enqueue_many([training_labels_data, training_img_data])

    labels, images = queue.dequeue_many(batch_size)

    discrimnator = create_discriminator(image_size)
    generated_img, params = generator_image(labels, image_size, batch_size)

    # Objective functions
    print("labels type: ", type(labels))
    print("images type: ", type(images))
    print("generated_img type: ", type(generated_img))
    loss_discrim = tf.reduce_mean(tf.log(discrimnator(tf.concat([labels, images], axis=3)) + 1e-12)) \
                   + tf.reduce_mean(tf.log(1.0 - discrimnator(tf.concat([labels, generated_img], axis=3)) + 1e-12))

    loss_gen_GAN = tf.reduce_mean(tf.log(1.0 - discrimnator(tf.concat([labels, generated_img], axis=3)) + 1e-12))
    loss_gen_l1 = tf.reduce_mean(tf.abs(images - generated_img))
    loss_gen = loss_gen_GAN + 100.0 * loss_gen_l1

    # Optimizer

    train_discriminator = tf.train.AdamOptimizer(.0002, beta1=.5).minimize(-loss_discrim,
                                                                           discrimnator.trainable_weights)

    train_generator = tf.train.AdamOptimizer(.0002, beta1=.5).minimize(loss_gen, var_list=[
        op for l in map(lambda x: x.trainable_weights, params) for op in l
    ])
