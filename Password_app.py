import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from pandas import Series
import flask
from flask import Flask, render_template, url_for, request


def word_to_char(word):
    return list(word)


model = pickle.load(open("final_model.pickle", "rb"))
vect = pickle.load(open('tfidf_password_strength.pickle', 'rb'))
app = flask.Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')

def preprocessing(text):
    # Find the length of character presented in password
    char_count = len(text)
    text: Series = pd.Series(text)
    # Find the length of number presented in password
    numerics = text.apply(lambda x: len(
        [str(x) for x in list(x) if str(x).isdigit()]))
    # Find the length of Alphabets presented in password
    alpha = text.apply(lambda x: len([x for x in list(x) if x.isalpha()]))
    # Find the length of punctuation presented in password
    specialSymbols = '!?%&.:~/\|,;<>-_+@#$%^&*(){}[]'
    punc = text.apply(lambda x: len(
        [x for x in list(x) if x in specialSymbols]))
    # Find the length of vowels presented in password
    vowels = ['a', 'e', 'i', 'o', 'u']
    vowels = text.apply(lambda x: len([x for x in list(x) if x in vowels]))
    # Find the length of consonants presented in password
    consonants = text.apply(lambda x: len(
        [x for x in list(x) if x not in vowels and x.isalpha()]))
    return char_count, numerics[0], alpha[0], punc[0], vowels[0], consonants[0]



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['password']
        length = np.array([text])
        print(length)
        data = vect.transform(length)
        print(data)
    my_prediction = model.predict(data)
    print(my_prediction)
    my_prediction=my_prediction[0]
    print(my_prediction)

    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run()
