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


path = 'C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\automationFinalData.xlsx'
df = red_Excel(path)

list_documents = []
list_lectura = []
list_triggers = []
list_fecha = []
historia = []
Id = []
ORIGEN = []
count = 0

for row_index,row in df.iterrows():
    lectura = ''
    temp = df[df['DOCUMENTO'] == row['DOCUMENTO']]

    for item2,row2 in temp.iterrows():
        lectura = lectura + ' ' + str(row2['LECTURA'])
    if row['DOCUMENTO'] not in list_documents:
        list_fecha.append(row['FECHA DE ESTUDIO'])
        list_documents.append(row['DOCUMENTO'])
        list_lectura.append(lectura)
        list_triggers.append(row2['TRIGGER'])
        historia.append(row2['historia'])
        Id.append(row2['Id'])
        ORIGEN.append(row2['ORIGEN'])

#print(lectura)

df2 = pd.DataFrame(list(zip(list_fecha,list_documents, list_lectura,list_triggers,historia,Id,ORIGEN)),
               columns =['FECHA DE ESTUDIO','DOCUMENTO', 'LECTURA','TRIGGER','historia','Id','ORIGEN'])

df2.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\AutomationV03.xlsx', index = False)