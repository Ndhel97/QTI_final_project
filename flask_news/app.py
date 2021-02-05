import pickle
import numpy as np
import string
from keras.models import load_model
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:password123@localhost:5432/postgres'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

model = load_model('models/model.h5')
tokenizer = pickle.load(open('models/tokenizer.pickle', 'rb'))


class NewsModel(db.Model):
    __tablename__ = 'news_tab'

    id = db.Column(db.Integer, primary_key=True)
    news = db.Column(db.String())
    news_category = db.Column(db.String())

    def __init__(self, news, news_category):
        self.news = news
        self.news_category = news_category

    def __repr__(self):
        return '<id {}>'.format(self.id)


def clean_text(text):
    st_words = stopwords.words()
    text = text.lower()
    text = text.translate(str.maketrans('-', ' '))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([w for w in text.split() if w not in st_words])
    return text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/', methods=['POST'])
def predict():
    global response, category
    if request.method == 'POST':
        text_raw = request.form['text']
        text = clean_text(text_raw)
        text = tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=256)
        result = model.predict([text])
        response = int(np.argmax(result))

        if response == 0:
            category = 'automobile'
        elif response == 1:
            category = 'entertainment'
        elif response == 2:
            category = 'politics'
        elif response == 3:
            category = 'science'
        elif response == 4:
            category = 'sports'
        elif response == 5:
            category = 'technology'
        elif response == 6:
            category = 'world'

        new_news = NewsModel(news=text_raw, news_category=category)
        db.session.add(new_news)
        db.session.commit()

    return render_template('results.html', prediction=response)


@app.route('/news_list/')
def list_news():
    return render_template('news_list.html', NewsModel=NewsModel.query.all())


@app.route('/news_list/delete/<news_id>', methods=['POST'])
def del_news(news_id):
    news = NewsModel.query.get_or_404(news_id)

    if request.method == 'POST':
        db.session.delete(news)
        db.session.commit()
    return render_template('news_list.html', NewsModel=NewsModel.query.all())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
