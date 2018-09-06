# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:06:55 2018

@author: Vishnu
"""

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
import numpy as np
from preprocess import clean_text, label

df = pd.read_csv('data.csv')

df= df.dropna()

df = df.replace({"sentiment": label})



df['content'] = df['content'].map(lambda x: clean_text(x))

vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['content'])

sequences = tokenizer.texts_to_sequences(df['content'])
data = pad_sequences(sequences, maxlen=100)

max_features = 20000
maxlen = 100
batch_size = 64

def model():
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    filepath = "model.h5"
    checkpointer = ModelCheckpoint(filepath)
    model.fit(data, np.array(df['sentiment']), 
              batch_size=batch_size, epochs=10,
              callbacks=[checkpointer])
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    return model, json_file

model()