from flask import Flask, flash, request, redirect, url_for, render_template
from PIL import Image
from pickle import load
from keras.models import load_model
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences
import urllib.request
import os
import numpy as np
import argparse

from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('caption.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        def extract_features(filename, model):
            try:
                image = Image.open(filename)
            except:
                print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
            image = image.resize((299,299))
            image = np.array(image)
            if image.shape[2] == 4: 
                image = image[..., :3]
            image = np.expand_dims(image, axis=0)
            image = image/127.5
            image = image - 1.0
            feature = model.predict(image)
            return feature

        def word_for_id(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None

        def generate_desc(model, tokenizer, photo, max_length):
            in_text = 'start'
            tex=""
            for i in range(max_length):
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=max_length)
                pred = model.predict([photo,sequence], verbose=0)
                pred = np.argmax(pred)
                word = word_for_id(pred, tokenizer)
                if word is None:
                    break
                in_text += ' ' + word
                if word == 'end':
                    break
            te=in_text.split()        
            for i in te:
                if(i=="start" or i=="end"):
                    continue
                tex=tex+i+" "    
             
            return tex

        max_length = 34
        tokenizer = load(open("Output/tokenizer.p","rb"))
        model = load_model('Output/model_9.h5')
        xception_model = Xception(include_top=False, pooling="avg")
        photo = extract_features(file, xception_model)
        img = Image.open(file)
        description = generate_desc(model, tokenizer, photo, max_length)
        print(description)
        flash('Image successfully uploaded and displayed below')
        image=Image.open(file)
        return render_template('caption.html', filename=filename,caption=description)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()