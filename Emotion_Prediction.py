# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:22:02 2018

@author: Vishnu
"""

from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from preprocess import clean_text, label

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

def prediction(text):
    text = clean_text(text)
    text = pd.Series(text)
    vocabulary_size = 20000
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    data = pad_sequences(sequences, maxlen=100)
    predicted = loaded_model.predict(data)
    predicted = int(predicted[0])
    prediction = [k for k, v in label.items() if v == predicted][0]
    return prediction