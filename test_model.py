import pickle

import numpy as np

import string
from keras.models import load_model

from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_model():
    model = load_model('models/model.h5')
    tokenizer = pickle.load(open('models/tokenizer.pickle', 'rb'))
    print('Model Loaded!!')
    return model, tokenizer


def clean_text(text):
    st_words = stopwords.words()
    text = text.lower()
    text = text.translate(str.maketrans('-', ' '))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([w for w in text.split() if w not in st_words])
    return text


def get_prediction(text, model, tokenizer):
    text = clean_text(text)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=256)
    res = model.predict([text])
    return print(int(np.argmax(res)))


test = 'Novel biosensors quickly detect coronavirus proteins, antibodies Scientists have developed new protein-based biosensors that glow when mixed with components of the novel coronavirus or specific COVID-19 antibodies, a breakthrough that could enable faster and more widespread testing for the disease.'
model, token = get_model()
get_prediction(test, model, token)
