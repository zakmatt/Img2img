import cv2
from keras.models import load_model as load_mdl
import keras.backend as K
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def is_filename_allowed(file_name):
    return '.' in file_name and \
        file_name.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def load_model(path):
    def generator_loss(y_true, y_pred):
        return K.mean(K.abs(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)
    return load_mdl(path, custom_objects={'generator_loss': generator_loss})

def normalize_pic(pic):
    return (pic / 127.5) - 1

def denormalize_pic(pic):
    return ((pic + 1) * 127.5).astype(np.uint8)

def process_image(label_img, model):
    label_img = normalize_pic(label_img)
    pic = model.predict(label_img)
    pic = denormalize_pic(pic)
    return pic

def load_image(path):
    image = cv2.imread(path).astype(np.float32)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    return image

def save_image(image, path):
    r, g, b = cv2.split(image)
    image = cv2.merge((b, g, r))
    cv2.imwrite(path, image)