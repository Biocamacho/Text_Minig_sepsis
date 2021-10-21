from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud
import string
from string import punctuation
import nltk
from nltk.corpus import stopwords
import plotly.express as px
from nltk.tokenize import sent_tokenize, word_tokenize
import itertools
import collections
from nltk import bigrams
import networkx as nx
from nltk.util import ngrams
import spacy
import es_core_news_sm
import gensim
from gensim import corpora
import time
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf



def clean_text(df):
    replacelist = ['conservadasecas','izquierdadiafragma','atrialsonda','glosectomiaradiografia','normaltransparencia','derram ',' erecho',' sign ']
    replace1 = ['conservada secas','izquierda diafragma','atrial sonda','glosectomia radiografia','normal transparencia','derrame',' derecho',' signo ']

    stop = stopwords.words('spanish')
    df['LECTURA'] = df['LECTURA'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
    df['LECTURA'] = df['LECTURA'].str.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u') 
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'\s+',' ', str(elem)))
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'\d+','', str(elem)))
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'[|]','', str(elem)))
    non_words = list(punctuation)
    non_words.extend(['¿', '¡', '‘', '’','/',')','(','\''])
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: ''.join([c for c in elem if c not in non_words]))

    for item in range(len(replacelist)): 
      df['LECTURA'] = df['LECTURA'].replace(replacelist[item],replace1[item])


def red_Excel(path):

    df = pd.read_excel(path)
    return df

def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    
    return count

path = 'C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\Automation.xlsx'
df = red_Excel(path)

count_class_0, count_class_1 = df.TRIGGER.value_counts()

# Divide by class
df_class_0 = df[df['TRIGGER'] == 0]
df_class_1 = df[df['TRIGGER'] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)


clean_text(df_test_under)
counter = counter_word(df_test_under.LECTURA)

#print(counter)

num_unique_words = len(counter)

# Split dataset into training and validation set 

train_size = int(df_test_under.shape[0]*0.8)

train_df = df_test_under[:train_size]
val_df = df_test_under[train_size:]

# split text and labels 
train_sentences = train_df.LECTURA.to_numpy()
train_labels = train_df.TRIGGER.to_numpy()
val_sentences = val_df.LECTURA.to_numpy()
val_labels = val_df.TRIGGER.to_numpy()

#print(train_labels.shape, val_sentences.shape)

#print(train_labels[12300:12315])

#Tokenize 
tokenizer  = Tokenizer(num_words = num_unique_words)
tokenizer.fit_on_texts(train_sentences)


# each word has unique index
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
#print(len(train_sequences))
val_sequences = tokenizer.texts_to_sequences(val_sentences)


print(train_sentences[10:15])
print(train_sequences[10:15])

# Pad the sequence to have the same lenght

# Max lenght of words in a sequence 
max_lenght = 150

train_padded = pad_sequences(train_sequences,maxlen=max_lenght,padding="post",truncating="post")
val_padded = pad_sequences(val_sequences,maxlen=max_lenght,padding="post",truncating="post")

print(train_padded.shape,val_padded.shape)


model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 32, input_length=max_lenght))

model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1, activation="sigmoid"))

print(model.summary())

loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(lr = 0.001)
metrics = ['accuracy']

model.compile(loss = loss, optimizer= optim, metrics=metrics)
model.fit(train_padded,train_labels, epochs=20, validation_data = (val_padded,val_labels), verbose=2)

predictions = model.predict(val_padded)
predictions = [1 if p> 0.5 else 0 for p in predictions]

#print(train_sentences[10900:10919])

print(val_labels[20:40])
print(predictions[20:40])

print(tf.math.confusion_matrix(
    train_labels, predictions, num_classes=None, weights=None, dtype=tf.dtypes.int32,
    name=None
))

# Regresion Lineal
# Regresion Logistica
# Red Neuronal de una capa
