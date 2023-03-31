import os
import numpy as np
import email
import re

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.snowball import SnowballStemmer

from flask import Flask, request, jsonify, render_template


def load_data():
    ham_path = os.path.join(os.getcwd(), 'server', 'data', 'easy_ham_2')
    spam_path = os.path.join(os.getcwd(), 'server', 'data', 'spam_2')

    x_spam = []
    x_ham = []

    for filename in os.listdir(ham_path):
        abs_path = os.path.join(ham_path, filename)
        with open(abs_path, 'r', encoding='utf8', errors='ignore') as f:
            x_ham.append(f.read())

    for filename in os.listdir(spam_path):
        abs_path = os.path.join(spam_path, filename)
        with open(abs_path, 'r', encoding='utf8', errors='ignore') as f:
            x_spam.append(f.read())

    return np.array(x_spam, dtype='object'), np.array(x_ham, dtype='object')


class EmailTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=False, convert_lower=True):
        self.strip_headers = strip_headers
        self.convert_lower = convert_lower

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_out = np.empty((x.shape), dtype=x.dtype)
        for i, e in enumerate(x):
            msg = email.message_from_string(e)

            if self.strip_headers:
                for k in msg.keys():
                    del msg[k]

            payload = msg.get_payload()

            if self.convert_lower:
                if isinstance(payload, list):
                    payload_str = ''.join(p.as_string().lower()
                                          for p in payload)
                    msg.set_payload(payload_str)
                else:
                    msg.set_payload(msg.get_payload().lower())

            x_out[i] = msg.as_string()

        return x_out


class EmailReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, replace_url=False, replace_number=False):
        self.replace_url = replace_url
        self.replace_number = replace_number

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_out = np.empty((x.shape), dtype=x.dtype)

        for i, e in enumerate(x):
            if isinstance(e, bytes):
                e = e.decode('ISO-8859-1')

            if self.replace_url:
                e = re.sub(r'http\S+|www\S+', 'URL', e)

            if self.replace_number:
                e = re.sub(r'\d+', 'NUMBER', e)

            e = e.lower()

            x_out[i] = e

        return x_out


class EmailStemmer(BaseEstimator, TransformerMixin):
    def __init__(self, use_porter=False):
        self.use_porter = use_porter

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_out = np.empty((x.shape), dtype=x.dtype)

        for i, e in enumerate(x):
            if self.use_porter:
                x_out[i] = SnowballStemmer('porter').stem(e)
            else:
                x_out[i] = SnowballStemmer('english').stem(e)

        return x_out


def main():

    x_spam, x_ham = load_data()
    y_spam = np.ones(x_spam.shape)
    y_ham = np.zeros(x_ham.shape)

    x = np.concatenate([x_spam, x_ham])
    y = np.concatenate([y_spam, y_ham])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    pipeline = make_pipeline(EmailTransformer(), EmailReplacer(
        replace_number=True, replace_url=True), EmailStemmer(), verbose=3)

    x_tr = pipeline.fit_transform(x)

    vectorizer = CountVectorizer(encoding='ISO-8859-1')
    vectorizer.fit(x_tr)

    x_train_tr = pipeline.transform(x_train)
    x_test_tr = pipeline.transform(x_test)

    x_train_tr = vectorizer.transform(x_train_tr)
    x_test_tr = vectorizer.transform(x_test_tr)

    gb = GradientBoostingClassifier()
    gb.fit(x_train_tr, y_train)

    # make a web server using flask and use the model to predict the spam or ham
    # and return the result to the user
    app = Flask(__name__)

    @app.route('/')
    # get the input from the user and return the result using the model to predict in json format
    # dont render the template or any html file
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        print('request received')
        if request.method == 'POST':
            print('request is post')
            message = request.form['message']
            print('message is: ', message)
            data = np.array([message])
            data_tr = pipeline.transform(data)
            data_tr = vectorizer.transform(data_tr)
            my_prediction = gb.predict(data_tr)
            print(my_prediction)

            return jsonify({'prediction': my_prediction[0]})

    return app


app = main()
