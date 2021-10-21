import io
import pandas as pd
from PyPDF2 import PdfFileReader
import urllib.request
import time
import csv
start_time = time.time()


def Extract_info():
  #IMPORT EXCEL FILE AS PANDAS DATA FRAME, THIS EXCEL FILE HAS ALL THE URLS THAT WE ARE READING AND COLLECTING THE INFORMATION
  excel_querys = pd.read_excel (r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\Query_Imagenología_2019JC.xlsx')
  excel_querys_example = excel_querys.head(n=100)
  lecturas_list = []
  df_reporte = pd.DataFrame()
  exception_list = []
  Id = 1

  for index,row in excel_querys.iterrows():

    try:
      url = row['reporte']
      estudio = row['estudio']
      origen = row['procedencia']

      resp = urllib.request.urlopen(url).read()

      f = io.BytesIO(resp)
      reader = PdfFileReader(f)
      contents=[]
      for i in range(reader.getNumPages()):
        contents.append(reader.getPage(i).extractText())
      f.close()

      lines = contents[0].split('\n')

      fecha = lines[11]
      documento = lines[3]

      string = ''
      for line in range(len(lines)):
        if line > 13 and line < len(lines) - 7:
          string = string + lines[line] + ' '

      
      string = string.replace('.  ','.')
      string = string.replace('€','')
      #print(string)

      data = {
          'Id': Id,
          'FECHA DE ESTUDIO': [fecha],
          'DOCUMENTO': [documento],
          'NOMBRE DE ESTUDIO': [estudio],
          'ORIGEN': [origen],
          'LECTURA': [string]
      }
      
      aux = pd.DataFrame(data)
      df_reporte = df_reporte.append(aux)
      print(Id)
      #print(len(df_reporte))
    except:
      
      exception_list.append(len(df_reporte))


    Id += 1

  #exception_list.to_txt(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\Exceptions.txt')

  df_reporte.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\Automation.xlsx', index = False)
  print(exception_list)

  print("Process finished --- %s seconds ---" % (time.time() - start_time))


  

Extract_info()


####################################PROCESO ADICIONAL PARA SEPARAR PACIENTES CON SEPSIS##################################################################


#def Select_sepsis_Patients():
# df = pd.read_csv('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\CSVs\\pacientes.csv')
# Sepsis_patients = df
# #print(Sepsis_patients.head())

# Documents_df = pd.read_csv('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\CSVs\\pacientesUciUce.csv')
# #print(Documents_df.head())

# Merge = Documents_df.merge(Sepsis_patients, left_on='historia', right_on='historia final')[['historia','historia final','inicio_sepsis','tipodocumento','documento','sepsis']]
# print(Merge.head())

# Merge['aux'] = Merge['tipodocumento'] + ' ' + Merge['documento']
# print(Merge.head())

# Merge.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\SepsisPatients.xlsx', index = False)