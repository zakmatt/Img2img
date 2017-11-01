from data import Data
import numpy as np
import tensorflow as tf

if __name__=='__main__':
    data = Data("../data/facades_data.h5")
    image_size = data.get_image_size()
    training_img_data = tf.placeholder(tf.float32, shape=[None] + image_size)
    sketch_img_data = tf.placeholder(tf.float32, shape=[None] + image_size)
    print(image_size)