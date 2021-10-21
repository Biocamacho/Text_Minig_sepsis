import pandas as pd
import numpy as np
import re
import datetime
from datetime import timedelta



def red_Excel(path):

    df = pd.read_excel(path)
    sepsis = pd.read_excel('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\SepsisPatients.xlsx')
    # MERGE DE DF
    final = sepsis.merge(df, left_on='aux', right_on='DOCUMENTO')
    return df,sepsis



path = 'C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\AutomationV03.xlsx'

df,sepsis = red_Excel(path)

itis=['colecistitis', 'pancreatitis', 'ascitis', 'cistitis', 'sinusitis', 'celulitis', 'mioscitis', 'fascitis', 
      'osteomielitis', 'mediastinitis', 'meningitis', 'diverticulitis', 'pielonefritis', 'mastoiditis', 'pansinusitis', 'gastroenteritis', 
      'bronquiolitis', 'peritonitis', 'endocarditis', 'otomastoiditis', 'pericarditis', 'pancolitis', 'mielitis', 'encefalitis', 'vasculitis', 
      'enterocolitis', 'colitis', 'proctitis', 'divertículitis', 'neumonitis','traqueobronquitis', 'bronquitis','bronquilitis', 'orquiepididimitis', 
      'leptomeningitis', 'cerebritis', 'tiroiditis', 'meningoencefalitis', 'colecisititis','neumonia','neumonía']
text_itis=''
for i in itis:
  text_itis=text_itis+'|'+i 
text_itis

pattern_negative='('+'(no |descarta|descartar)(\w*\s)+'+'(\w*\s){0,2}'+'(sepsis|shock séptico'+text_itis+'))'
pattern_positive='((\w*\s*){0,2}sepsis|shock séptico'+text_itis+')'

print(pattern_negative)
print(pattern_positive)

pattern_positive_WN='(?<!no[ ])(?<!descarta[ ])(?<!descartar[ ])((\w*\s){0,2})(sepsis|shock séptico'+text_itis+')'


df['count_sepsis_words_negative']=df['LECTURA'].str.count(r''.join(pattern_negative))

dfnegativa = df[df['count_sepsis_words_negative']>0]
negativa = dfnegativa.DOCUMENTO.tolist()




count = 0
for item,row in df.iterrows():
    for word in itis:
        if row.DOCUMENTO not in negativa:
            df.at[count, word] = str(row.LECTURA.lower()).count(word) 
        #print(type(word),type(row.tokens_clean))
    count +=1
# print(count)


df.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\itis2.xlsx', index = False)
