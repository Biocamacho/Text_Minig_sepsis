import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from string import punctuation
import matplotlib.pyplot as plt




def red_Excel(path):

    df = pd.read_excel(path)
    return df


path = 'C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\AutomationV01.xlsx'
df = red_Excel(path)





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

clean_text(df)

#df.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\AutomationV2.xlsx', index = False)

x = df.LECTURA.values
y = df.TRIGGER.values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20, random_state = 32)

vectorizer = CountVectorizer()
vectorizer.fit(x_train)

x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)

classifier = LogisticRegression(max_iter = 1000)
classifier.fit(x_train, y_train)

score = classifier.score(x_train, y_train)

print("Accuracy: ", score)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred, labels=df.TRIGGER.unique())
df_cm = pd.DataFrame(cm, index=df.TRIGGER.unique(), columns=df.TRIGGER.unique())
print(df_cm)

# t = 'indicación  años cateterismo cardiacoexpansión pulmonar bilateral aumentada moderados signos atrapamiento aireparénquima pulmonar signos consolidación lesión inflamatoria infecciosa expansivadistribución vasculatura pulmonar normalmoderada cardiomegalia crecimiento cámaras izquierdasaorta elongada cambios ateromatosismediastino centradodiafragma ángulos libres signos derrameespondilosis osteopenia columna dorsal técnica utilizada mas  kv  dosis mgy '

# vect_lect = vectorizer.transform(np.array([t]))

# prediction = classifier.predict(vect_lect)
# #print('this is the prediction: ',prediction[0])

score = classifier.score(x_test, y_test)

print("Accuracy: ", score)