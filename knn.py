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
from collections import Counter
from pylab import rcParams
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC


# np.set_printoptions(precision = 4,suppress=True)
# rcParams['figure.figsize'] = 7,4
# plt.style.use('seaborn-whitegrid')

#------------------------------------------SPLIT YOUR DATASET---------------------------------------------


def red_Excel(path):

    df = pd.read_excel(path)
    return df

def clean_text(df):
    replacelist = ['conservadasecas','izquierdadiafragma','atrialsonda','glosectomiaradiografia','normaltransparencia','derram ',' erecho',' sign ']
    replace1 = ['conservada secas','izquierda diafragma','atrial sonda','glosectomia radiografia','normal transparencia','derrame',' derecho',' signo ']

    stop = stopwords.words('spanish')   
    non_words = list(punctuation)
    non_words.extend(['¿', '¡', '‘', '’','/',')','(','\''])

    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'\s+',' ', str(elem)))
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'\d+','', str(elem)))
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'[|]','', str(elem)))
    df['LECTURA'] = df['LECTURA'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))



    df['LECTURA'] = df['LECTURA'].apply(lambda elem: ''.join([c for c in elem if c not in non_words]))

    for item in range(len(replacelist)): 
      df['LECTURA'] = df['LECTURA'].replace(replacelist[item],replace1[item])

path = 'C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\AutomationV3.xlsx'
df = red_Excel(path)
clean_text(df)

x = df.LECTURA.values
y = df.TRIGGER.values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20, random_state = 32)

vectorizer = CountVectorizer()
vectorizer.fit(x_train)

x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)
#----------------------------------BUILDING AND TRAINING YOUR MODEL WITH TRAINING DATA-------------------------

# lr_model = SVC(gamma='auto')
# lr_model.fit(x_train,y_train)
lr_model = KNeighborsClassifier(n_neighbors= 20)
lr_model.fit(x_train,y_train)

predictions = lr_model.predict(x_test)

print(metrics.classification_report(y_test,predictions))


#print(lr_model)
