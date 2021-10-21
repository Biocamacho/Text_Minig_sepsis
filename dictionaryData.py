import pandas as pd
import numpy as np
import re
import datetime
from datetime import timedelta



def red_Excel(path):

    df = pd.read_excel(path)
    sepsis = pd.read_excel('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\SepsisPatients.xlsx')
    # MERGE DE DF
    final = df.merge(sepsis, right_on='aux', left_on='DOCUMENTO')
    return final,sepsis



path = 'C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\FinalDATA6.xlsx'

df,sepsis = red_Excel(path)

print(df.head())

#df.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\test3.xlsx', index = False)
df_test = df[df['DOCUMENTO']=='CC 32401246']
fecha = df_test['Id'].tolist()

for item in fecha:
    #print(df.loc[item,'FECHA DE ESTUDIO'])
    df.at[df.Id == item,'FECHA DE ESTUDIO'] = '2020-04-13'

print(df[df['DOCUMENTO']=='CC 32401246'])

sepsis_list = sepsis.aux.tolist()
df_final = pd.DataFrame()
count =0

lectura = []
documento = []
sep = []

list_sepsis = sepsis.aux.tolist()
list_fecha = df['inicio_sepsis'].tolist()
list_documents = df.DOCUMENTO.tolist()
list_lectura = df.LECTURA.tolist()
fecha = df['FECHA DE ESTUDIO'].tolist()
list_triggers = df['sepsis'].tolist()
count_list = []

count = 0
for index,row in sepsis.iterrows():
    #print(count)
    try:
        

        if int(row['inicio_sepsis'][:4]) == 2061:
            row['inicio_sepsis'] = row['inicio_sepsis'].replace('61','16')
            #print(int(row['inicio_sepsis'][:4]))
        elif int(row['inicio_sepsis'][:4]) == 2062:
            row['inicio_sepsis'] = row['inicio_sepsis'].replace('62','17')
        elif int(row['inicio_sepsis'][:4]) == 2063:
            row['inicio_sepsis'] = row['inicio_sepsis'].replace('63','18')
        elif int(row['inicio_sepsis'][:4]) == 2064:
            row['inicio_sepsis'] = row['inicio_sepsis'].replace('64','19')
        else:
            row['inicio_sepsis'] = row['inicio_sepsis'].replace('65','20')

        #print(1)

        #fechaEstudio = datetime.datetime(int(row['FECHA'][:4]),int(row['FECHA'][5:7]),int(row['FECHA'][8:10]))
        #print(1)
        inicio_sepsis = datetime.datetime(int(row['inicio_sepsis'][:4]),int(row['inicio_sepsis'][5:7]),int(row['inicio_sepsis'][8:10]))
        #print(1)
        #fin_sepsis = inicio_sepsis - timedelta(hours=24)
        #print(1)
        #print(fechaEstudio,inicio_sepsis,fin_sepsis)

        df_test = df[df['DOCUMENTO']==row['aux']]
        fecha = df_test['Id'].tolist()

        for item in fecha:
            #print(df.loc[item,'FECHA DE ESTUDIO'])
            df.at[df.Id == item,'FECHA DE ESTUDIO'] = inicio_sepsis

        # if ( fechaEstudio >= inicio_sepsis) and (fechaEstudio <=fin_sepsis):
        #     count_list.append(1)
        #     print(1)
        # else:
        #     count_list.append(0)
    except:
        print(1)
        #count_list.append(0)




# df_final = pd.DataFrame(list(zip(fecha,list_documents, list_lectura,list_triggers,list_fecha,count_list)),
#                  columns =['FECHA','DOCUMENTO', 'LECTURA','TRIGGER','FECHASEPSIS','COUNT'])

df.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\FinalDATA6.xlsx', index = False)