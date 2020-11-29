
import os
import sys

# Flask
from flask import Flask, request, render_template

# Some utilites
import numpy as np
from model_files.models import model_predict
from PIL import Image
import base64


# Classes

classes = {
0 : "Black Bin",
1 : "Blue Bin",
2 : "Green Bin",
3 : "Regular Garbage"
}


# Declare a flask app
app = Flask("Bin_Predictor")
app.secret_key = "dharnashukla94"


uploads_dir = "static/upload"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)



@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', result="No image selected")

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', result="Please upload an image")


        file.save(os.path.join(uploads_dir, file.filename))
        location = os.path.join(uploads_dir, file.filename)

        img = Image.open(location)

        # Make prediction
        preds = model_predict(img)
        pred_proba = "{:.3f}".format(np.amax(preds))
        max_index = np.argmax(preds, axis=1)

        with open(location, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        os.remove(location)
        return render_template('index.html', result=classes[max_index[0]], image=encoded_image)


    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 8080)
