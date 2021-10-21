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
    
    return df


def get_tokens(final):
    #TOKENIZACION
  
    final['tokens'] = final['LECTURA'].apply(lambda x: word_tokenize(str(x)))
    final['tokens_clean']=final['tokens']
    final['tokens_clean']=final['tokens_clean'].apply(lambda text: [word for word in text if word.isalnum() and len(word)>1])

    return final

def bigrams(final):
    #FRECUENCIA DE BIGRAMAS
    terms_bigram = [list(nltk.bigrams(t)) for t in final.tokens_clean]

    bigramsList = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigramsList)
    bigram_counts.most_common(15)

    bigram_df = pd.DataFrame(bigram_counts.most_common(200),columns=['bigram', 'count'])

    d = bigram_df.set_index('bigram').T.to_dict('records')
    return d

def counter_word(text_col):
    count = Counter()
    for text in text_col:
        for word in text:
            count[word] += 1
    
    return count



path = 'C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\test3.xlsx'
df = pd.read_excel(path)
sepsis = pd.read_excel('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\SepsisPatients.xlsx')
sepsis = sepsis[sepsis['sepsis'] == 1]
list_sepsis = sepsis.aux.tolist()
list_documents = df.DOCUMENTO.tolist()
list_lectura = df.LECTURA.tolist()
fecha = df['FECHA DE ESTUDIO'].tolist()
list_triggers = []
historia = df['historia'].tolist()
Id = df['Id'].tolist()
ORIGEN = df['ORIGEN'].tolist()


for item in list_documents:
    if item in list_sepsis:
        list_triggers.append(1)
    else:
        list_triggers.append(0)

    
# print(1)

df = pd.DataFrame(list(zip(fecha,list_documents, list_lectura,list_triggers,historia,Id,ORIGEN)),
               columns =['FECHA DE ESTUDIO','DOCUMENTO', 'LECTURA','TRIGGER','historia','Id','ORIGEN'])



#df2 = clean_text(df)
#df2 = get_tokens(df2)
df.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\automationFinalData.xlsx', index = False)



# #---------------------------------UNIGRAMAS--------------------------------------------------

#list_tokens = df2[df2['TRIGGER'] == 0].tokens_clean.tolist()
# list_tokens = df2[df2['TRIGGER'] == 1].tokens_clean.tolist()
#list_tokens = df2.tokens_clean.tolist()

# unique_words = []
# counter = counter_word(list_tokens)

# for key in counter.keys():
#     if key not in unique_words and counter[key] > 1000:
#         unique_words.append(key)



# print((unique_words))


# print(2)

# for col in unique_words:

#     df2[col] = 0


# print(3)



# count = 0
# for item,row in df.iterrows():
#     for word in unique_words:
#         df2.at[count, word] = row.tokens_clean.count(word) 
#         #print(type(word),type(row.tokens_clean))
#     count +=1
# # print(count)

# print(df2.head())
# df2.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\TensorflowDataUnigramsUnder1.xlsx', index = False)


#--------------------------------BIGRAMAS---------------------------------------------------
# unique_words = []
# bigrams = bigrams(df2)
# #bigrams = bigrams(df2[df2['TRIGGER'] == 0])
# print(bigrams)

# for key in bigrams[0].keys():
#     unique_words.append(key)

# for col in unique_words:

#     df2[col] = 0

# count = 0
# f = 0
# for column,row in df.iterrows():
#     for word in unique_words:
#         for item in word:
#           count  += row.tokens_clean.count(item) 
#         if count == 1:
#             df2.at[f, word] = 0
#         else:
#             df2.at[f, word] = int(count/2)
#         count = 0
#     f += 1

# print(df2.head())
# df2.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\TensorflowDataBigrams2.xlsx', index = False) 


