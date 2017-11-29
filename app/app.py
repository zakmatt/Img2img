from flask import Flask, request, redirect, url_for, render_template, send_from_directory, json
import logging
from utils.utils_methods import is_filename_allowed, process_image, save_image, load_model
import os
import time
from werkzeug.utils import secure_filename


MODEL_PATH = "../model/training_results/batch_10/pix2pix_g_epoch_100.h5"
IMAGES_FOLDER = "./static/images"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.logger.debug('Starting application')

model = load_model(MODEL_PATH)
logging.info('Model loaded.')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('main.html')

@app.route('/processImage', methods=['POST'])
def process():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if not file.filename:
        return redirect(request.url)

    if file and is_filename_allowed(file.filename):

        logging.info('Processing an image.')
        #generated_image = process_image(file)
        #save_image(generated_image, './static/images/generated_image.jpg')
        file.save('./static/images/generated_image.jpg')

        generated_image = 'static/images/generated_image.jpg' + str('?%s' % time.time())

    return json.dumps({
        'status': 'OK',
        'generated_image': generated_image
    })

app.run(port=5000)