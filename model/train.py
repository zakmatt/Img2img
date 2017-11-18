from data import Data
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


from generator import Generator
from discriminator import Discriminator

from keras.layers import Input, Concatenate
from keras.models import Model
import keras.backend as K

# Progress bar
from tqdm import tqdm

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

def create_generator(image_size, batch_size=64):
    layer_params = {
        'ngf': 64,
        'filter_size': 4,
        'input_channels': image_size[2],
        'img_width': image_size[0],
        'img_height': image_size[1]
    }

    generator = Generator(None, layer_params)
    model, params = generator.create_model(batch_size=batch_size)
    return model


def create_gan(generator, discriminator, image_size):
    inputs = Input(image_size)
    x_generator = generator(inputs)

    merged = Concatenate(axis=-1)([inputs, x_generator])
    discriminator.trainable = False
    x_discriminator = discriminator(merged)

    model = Model(inputs, [x_generator, x_discriminator])

    return model

'''
def create_generator(labels, image_size, batch_size=64):
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
'''

def gan_loss(y_true,y_pred):
    return K.mean(K.binary_crossentropy(y_pred,y_true), axis=-1)

def generator_loss(y_true,y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)

if __name__ == '__main__':
    batch_size = 64
    number_of_epochs = 200

    # Training dataset
    train_data = Data('../data/facades/', 'train')
    image_size = train_data.get_image_size()
    dataset_size = train_data.data_amount()
    number_of_batches = dataset_size // batch_size
    print(number_of_batches)
    train_labels, train_targets = train_data.get_data()

    # Validation dataset

    val_data = Data("../data/facades/", "val")
    val_labels, val_targets = val_data.get_data()

    # Model

    generator = create_generator(image_size)
    discriminator = create_discriminator(image_size)

    gan = create_gan(generator, discriminator, image_size)

    generator.compile(loss=generator_loss, optimizer='RMSprop')
    gan.compile(loss=[generator_loss, gan_loss], optimizer='RMSprop')
    discriminator.trainable = True
    discriminator.compile(loss=gan_loss, optimizer='RMSprop')

    gan_losses = []
    discrim_losses = []

    y_dis = np.zeros((2 * batch_size, 30, 30, 1))
    y_dis[:batch_size] = 1.0

    for epoch in range(number_of_epochs):
        print('-' * 15, 'Epoch: {0}'.format(epoch), '-' * 15)
        for _ in tqdm(range(number_of_batches)):
            random_indices = np.random.randint(0, dataset_size, size=batch_size)
            batch_labels = train_labels[random_indices]
            batch_targets = train_targets[random_indices]

            #batch_labels_2 = np.tile(batch_labels, (2, 1, 1, 1))

            y_dis = np.zeros((2 * batch_size, 30, 30, 1))
            y_dis[:batch_size] = 1.0
            generated_images = generator.predict(batch_labels)

            # Default is concat first dimention
            #concat_pic = np.concatenate((batch_targets, generated_images))

            pred_true = np.concatenate((batch_targets, batch_labels), axis=-1)
            pred_false = np.concatenate((generated_images, batch_labels), axis=-1)
            discriminator_input = np.concatenate((pred_true, pred_false))

            dis_out = discriminator.predict(discriminator_input)

            d_loss = discriminator.train_on_batch(discriminator_input, y_dis)

            random_indices = np.random.randint(1, dataset_size, size=batch_size)
            batch_labels = train_labels[random_indices]
            batch_targets = train_targets[random_indices]
            y_gener = np.ones((batch_size, 30, 30, 1))
            discriminator.trainable = False
            g_loss = gan.train_on_batch(batch_labels, [batch_targets, y_gener])
            discriminator.trainable = True
        gan_losses.append(g_loss)
        discrim_losses.append(d_loss)

    '''
    import cv2
    super_file = ((train_labels[0]+1)*127.5).astype(np.uint8)
    r, g, b = cv2.split(super_file)
    super_file = cv2.merge((b, g, r))
    cv2.imwrite('super_file.jpg', super_file)
    '''