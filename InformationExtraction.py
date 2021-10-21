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


class GetData:
  #LECTURA EXCELES

  def red_Excel(self,path):

    df = pd.read_excel(path)
    sepsis = pd.read_excel('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\SepsisPatients.xlsx')
    # MERGE DE DF
    final = sepsis.merge(df, left_on='aux', right_on='DOCUMENTO')
    return df

class negativeData:


  def clean_text(self,df):
    replacelist = ['conservadasecas','izquierdadiafragma','atrialsonda','glosectomiaradiografia','normaltransparencia']
    replace1 = ['conservada secas','izquierda diafragma','atrial sonda','glosectomia radiografia','normal transparencia']

    stop = stopwords.words('spanish')
    df['LECTURA'] = df['LECTURA'].str.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u') 
    #df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'(\&gt\;)|(\&lt\;)','', elem))
    df['LECTURA'] = df['LECTURA'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'\s+',' ', elem))
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'\d+','', elem))
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: re.sub(r'[|]','', elem))
    ## Eliminar signos de puntuación '[!#?,.:";]'
    #df[text_field] = df[text_field].apply(lambda elem: re.sub(r"""[‘’]""",' ', elem))
    non_words = list(punctuation)
    non_words.extend(['¿', '¡', '‘', '’','/',')','('])
    df['LECTURA'] = df['LECTURA'].apply(lambda elem: ''.join([c for c in elem if c not in non_words]))

    for item in range(len(replacelist)): 
      df['LECTURA'] = df['LECTURA'].replace(replacelist[item],replace1[item])
      #print(replace1[item])


  def text_extract_for_wordcloud(self,df):

    text = ' '.join(txt for txt in df.LECTURA)
    text_list = text.split()
    return text

  def wordcloud(self,text):

    wordcloud = WordCloud(width=1280, height=690).generate(text)
    plt.figure(figsize=(15,10))
    plt.imshow(wordcloud) 
    plt.axis("off")
    print(plt.show())

  def get_tokens(self,final):
    #TOKENIZACION
  
    final['tokens'] = final['LECTURA'].apply(lambda x: word_tokenize(x))
    final['tokens_clean']=final['tokens']
    final['tokens_clean']=final['tokens_clean'].apply(lambda text: [word for word in text if word.isalnum() and len(word)>1])

  def frecuency_of_words_tokens(self,final):
    #FRECUENCIA DE PALABRAS
    all_words = list(itertools.chain(*final['tokens_clean']))
    counts_words = collections.Counter(all_words)
    mostcommon_words=counts_words.most_common(35)
    #print(mostcommon_words)
    return mostcommon_words

  def bigrams(self,final):
    #FRECUENCIA DE BIGRAMAS
    terms_bigram = [list(nltk.bigrams(t)) for t in final.tokens_clean]

    bigramsList = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigramsList)
    bigram_counts.most_common(15)

    bigram_df = pd.DataFrame(bigram_counts.most_common(50),columns=['bigram', 'count'])

    d = bigram_df.set_index('bigram').T.to_dict('records')
    return d

  def bigrams_graph(self,d):

    G = nx.Graph()
    # Conecciones entre los nodos
    #, weight=(v * 100)
    for k, v in d[0].items():
      G.add_edge(k[0], k[1])

    #G.add_node("", weight=100)
    fig, ax = plt.subplots(figsize=(25, 15))
    pos = nx.spring_layout(G, k=2)
    nx.draw_networkx(G, pos,font_size=10,width=2,edge_color='black',node_color='red',with_labels = False,ax=ax)
    for key, value in pos.items():
      x, y = value[0], value[1]+.035
      ax.text(x, y, s=key, bbox=dict(facecolor='red', alpha=0.25),horizontalalignment='center', fontsize=13)    
    print(plt.show())


  def lematizacion(self,final):
    
    nlp = es_core_news_sm.load()
    #print(final.LECTURA[0])
    d=nlp(final.LECTURA[0])
    t=[[tok.text,tok.lemma_,tok.pos_,tok.ent_type_] for tok in d]
    #print(t)
    # for noun in d.noun_chunks:
    #   print(noun.text)

    final['lemmas'] = final.LECTURA.apply(lambda text: [tok.lemma_ for tok in nlp(text)])
    #print(final.head())


  def celan_lemmas(self,final):
    allWords=[]
    for l in final.lemmas:
      allWords.extend(l)
    wordDist = nltk.FreqDist(allWords)
    
    #print(wordDist.most_common(15))

  def bag_of_words(self,df):
    words=[]
    for w in df.lemmas:
      words.append(w)
    print(words)

class PositiveData:


  def clean_text(self,df):

      replacelist = [   'conservadasecas','izquierdadiafragma','atrialsonda',
                        'glosectomiaradiografia','normaltransparencia',
                        'derram ',' erecho',' sign ']
      replace1 = [     'conservada secas','izquierda diafragma','atrial sonda',
                        'glosectomia radiografia','normal transparencia',
                        'derrame',' derecho',' signo ']

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
      



  def text_extract_for_wordcloud(self,df):

    text = ' '.join(txt for txt in df.LECTURA)
    text_list = text.split()
    return text_list

  def wordcloud(self,text):

    wordcloud = WordCloud(background_color='white',width=1280, height=690).generate(str(text))
    plt.figure(figsize=(15,10))
    plt.imshow(wordcloud) 
    plt.axis("off")
    print(plt.show())

  def get_tokens(self,final):
    #TOKENIZACION
  
    final['tokens'] = final['LECTURA'].apply(lambda x: word_tokenize(str(x)))
    final['tokens_clean']=final['tokens']
    final['tokens_clean']=final['tokens_clean'].apply(lambda text: [word for word in text if word.isalnum() and len(word)>1])

  def frecuency_of_words_tokens(self,final):
    #FRECUENCIA DE PALABRAS
    all_words = list(itertools.chain(*final['tokens_clean']))
    counts_words = collections.Counter(all_words)
    mostcommon_words=counts_words.most_common(35)
    return mostcommon_words

  def bigrams(self,final):
    #FRECUENCIA DE BIGRAMAS
    terms_bigram = [list(nltk.bigrams(t)) for t in final.tokens_clean]

    bigramsList = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigramsList)
    bigram_counts.most_common(15)

    bigram_df = pd.DataFrame(bigram_counts.most_common(50),columns=['bigram', 'count'])

    d = bigram_df.set_index('bigram').T.to_dict('records')
    return d

  def bigrams_graph(self,d):

    G = nx.Graph()
    # Conecciones entre los nodos
    #, weight=(v * 100)
    for k, v in d[0].items():
      G.add_edge(k[0], k[1])

    #G.add_node("", weight=100)
    fig, ax = plt.subplots(figsize=(15, 15))
    pos = nx.spectral_layout(G,scale=2)
    nx.draw_networkx(G, pos,node_size=60,font_size=8,edge_color='black',node_color='red',with_labels = False,ax=ax)
    for key, value in pos.items():
      x, y = value[0], value[1]+.035
      ax.text(x, y, s=key, bbox=dict(facecolor='red', alpha=0.25),horizontalalignment='center', fontsize=13)    
    print(plt.show())


  def lematizacion(self,final):
    
    nlp = es_core_news_sm.load()
    d=nlp(final.LECTURA[0])
    t=[[tok.text,tok.lemma_,tok.pos_,tok.ent_type_] for tok in d]

    final['lemmas'] = final.LECTURA.apply(lambda text: [tok.lemma_ for tok in nlp(text)])
    ejemploOracion = nlp(final.LECTURA[1])
    spacy.displacy.render(ejemploOracion, style='dep', jupyter=False, options={'distance': 90})
    print(plt.show())



  def celan_lemmas(self,final):
    allWords=[]
    for l in final.lemmas:
      allWords.extend(l)
    wordDist = nltk.FreqDist(allWords)
    
    print(wordDist.most_common(15))

  def bag_of_words(self,df):
    words=[]
    for w in df.lemmas:
      words.append(w)
    #print(words)


# -----------------------------------------------GET DATAFRAME--------------------------------------------------
# In this part i only get the data from the patiens with sepsis, to know how 
# please check the function red_excel were a merge is being performed. 
path = 'C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\automationV2.xlsx'
data = GetData()
df_sepsis = data.red_Excel(path)
#df_sepsis = df_sepsis.head(10)


#------------------------------------------------Perform the data extraction from non estructure texts and also clean the data----------------------------------------------------
#                                                                   (ONLY FOR POSITIVE EXPRESSIONS)
positive = PositiveData()

# EXTRACT TEXT FOR WORDCLOUD
positive.clean_text(df_sepsis)
#print(len(df_sepsis))
Wordcloud_text = positive.text_extract_for_wordcloud(df_sepsis)

# WORDCLOUD
#positive.wordcloud(Wordcloud_text)

# TOKENIZACION
positive.get_tokens(df_sepsis)
# print(df_sepsis.head(10))
common_words = positive.frecuency_of_words_tokens(df_sepsis)
print(common_words)

# df_sepsis.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\Tokens.xlsx', index = False)

# BIGRAMAS GRAPH 
# bigrams_consume = positive.bigrams(df_sepsis)
# positive.bigrams_graph(bigrams_consume)

# #LEMATIZACION
positive.lematizacion(df_sepsis)
positive.celan_lemmas(df_sepsis)

df_sepsis.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\Lemmas.xlsx', index = False)

# print(df_sepsis.head(10))


# # BAG OF WORDS
# positive.bag_of_words(df_sepsis)


#------------------------------------------------Perform the data extraction from non estructure texts and also clean the data----------------------------------------------------
#                                                                   (ONLY FOR NEGATIVE EXPRESSIONS)
# negative = negativeData()

# EXTRACT TEXT FOR WORDCLOUD
# negative.clean_text(df_sepsis)
#Wordcloud_text = negative.text_extract_for_wordcloud(df_sepsis)

# WORDCLOUD
#negative.wordcloud(Wordcloud_text)

# TOKENIZACION
# negative.get_tokens(df_sepsis)
# common_words = negative.frecuency_of_words_tokens(df_sepsis)
#print(common_words)

# BIGRAMAS GRAPH 
# bigrams_consume = negative.bigrams(df_sepsis)
# negative.bigrams_graph(bigrams_consume)

# #LEMATIZACION
# negative.lematizacion(df_sepsis)
# negative.celan_lemmas(df_sepsis)


# # BAG OF WORDS
# negative.bag_of_words(df_sepsis)


# from nltk.stem import PorterStemmer 
# from nltk.stem.snowball import SnowballStemmer ## Para poder hacerlo con idiomas diferentes al inglés
# stemmer_spanish = SnowballStemmer("spanish")
# df_sepsis['tokens_stem'] = df_sepsis['tokens'].apply(lambda text: [s for s in [stemmer_spanish.stem(i) for i in text] if s.isalpha() and len(s) > 1])
# print(df_sepsis.head(5))














