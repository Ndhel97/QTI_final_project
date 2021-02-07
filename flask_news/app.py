import pickle
import numpy as np
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
model = load_model('models/model.h5')
tokenizer = pickle.load(open('models/tokenizer.pickle', 'rb'))


def clean_text(text):
    st_words = open('./models/english', 'r').read().split()
    text = text.lower()
    text = text.translate(str.maketrans('-', ' '))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([w for w in text.split() if w not in st_words])
    return text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global response
    if request.method == 'POST':
        text = request.form['text']
        text = clean_text(text)
        text = tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=256)
        result = model.predict([text])
        response = int(np.argmax(result))

    return render_template('results.html', prediction=response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
