# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


def model_predict(img):
    img = img.resize((150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)

    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    MODEL_PATH = 'model_files/VGG19.h5'

    # Load trained model

    model = load_model(MODEL_PATH)
    model._make_predict_function()
    print('Model loaded. Start serving...')


    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds
