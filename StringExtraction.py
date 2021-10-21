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
    sepsis = pd.read_excel('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\SepsisPatients.xlsx')
    # MERGE DE DF
    final = sepsis.merge(df, left_on='aux', right_on='DOCUMENTO')
    return df



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




df = pd.read_excel('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\AutomationV03.xlsx')
clean_text(df)
#print(df.head())
#df = df[df['TRIGGER'] == 1]
regex = '/(\binfeccion|\bsepsis|\bsepticemia|\bneumonia|\bchoque|shock|\w*itis)+/g'
list1 = []
list2 = []
unique = []

x = '  opacidad hemitórax izquierdo predominio basal derrame poder bbronquiolitis componenitis parenquimatoso quede opacificado derrameimagen redondeada densas basal derecha puede corresponder neumonía redondasilueta cardíaca evaluado sumación densidadescatéter inserción periférica ingresando extremidad superior derecha extremo distal confluente cavo lateralpaciente queda rotada izquierda derecha indicacion descartar sinusitis tecnica cortes axiales simples topografía senos paranasales reconstrucciones multiplanares hallazgoslos senos paranasales muestran adecuada neumatización desarrollo edad no visualizan niveles hidroaéreos zonas engrosamiento mucoso indiquen compromiso inflamatorio tampoco imágenes sugestivas pólipos quistes retenciónel reborde óseo senos paranasales encuentra adecuadamente definido lineas fracturala pirámide nasal conserva altura morfología normalseptum nasal desviado hacia izquierdatejidos blandos intranasales espesor normalconclusión tomografía simple senos paranasales hallazgos patológicos en tomografía multidetector realizaron cortes axiales después administrar medio contraste intravenoso  cchallazgosregión supraclavicular normalen mediastino encuentro masas adenopatíaslas estructuras vasculares pulmonares opacifican adecuadamente medio contraste observarse imágenes hipodensas defecto indiquen presencia tromboembolismo pulmonarcavidades cardíacas aumentadas tamañono derrame pericárdicose encuentra derrame pleural bilateral loculado atelectasias pasivas asociadasen parénquima pulmonar neumotórax tampoco signos consolidación masas nódulosengrosamientos pleuro septales basales bilateraleslo visible abdomen superior hallazgos patológicosconclusiónderrame pleural bilateral atelectasias pasivas asociadas elementos monitoreo externohilios pulmonares congestivosinfiltrados mixtos predominio intersticial regiones centrales bibasalesno atelectasia mayor neumotórax técnica se realizaron cortes axiales bases pulmonares sínfisis púbica luego administración medio contraste intravenoso ml oral ml verificándose adecuada función renal previamenteinforme se observa derrame pleural bilateral atelectasias pasivas asociadasel hígado tamaño forma posición normal lesiones focales edema periportal no dilatación paredes vía biliar intra extrahepáticavesícula biliar ausente antecedente quirúrgico previopáncreas cambios inflamatorios lesiones neoplásicasbazo normalglándulas suprarrenales masas nóduloslos riñones tamaño posición normal dilatación cavidades colectores tampoco masas cálculos pequeños quistes simples renales bilaterales mayores  mmestructuras vasculares calibre normalno adenopatías retroperitonealeslas asas intestinales opacifican adecuadamente medio contraste signos obstructivoslíquido libre cavidad abdominal predominio espacio morrison gotera parietocólica derecha región pélvicavejiga distendida paredes delgadas lesiones interiorútero anexos normales catéter central izquierdo punta topografía aurícula derecha sonda esofagogástrica llama atención disposición sonda aspecto proximal correlacionar estrictamente datos clínicosno ensanchamiento mediastínico no puede valorar adecuadamente índice cardiotorácico pues definen contornos cardíacos basalessignos derrame  engrosamiento pleural bilateral no imagen neumotóraxno descarta atelectasia  consolidación basal sobreposición imágenes paciente queda rotada izquierda derecha catéter picc ingresando extremidad izquierda extremo distal confluente cavoatrial opacidad basal izquierda siendo difícil determinar si corresponde sumación densidades rotación anteriormente descrita componente pleuralproceso intersticial ambos campos pulmonaressilueta cardíaca aumentada tamaño tener cuenta magnificación ser proyección decúbito'

#print([re.findall("(infeccion|sepsis|septicemia|neumonia|choque|shock|neumonitis|bronquiolitis|bronquitis)", x)])


for item, row in df.iterrows():
    #list2.append(([re.findall("(\binfeccion|\bsepsis|\bsepticemia|\bneumonia|\bchoque|\bshock|\w*itis)", row['LECTURA'])]))
    list1.append(len([re.findall("""(infeccion|sepsis|septicemia|neumonia|choque|shock|neumonitis|bronquiolitis|bronquitis|colecistitis|pancreatitis|ascitis
                                    |cistitis|sinusitis|celulitis|mioscitis|fascitis|osteomielitis|mediastinitis|meningitis|diverticulitis|pielonefritis|mastoiditis
                                    |pansinusitis|gastroenteritis|bronquiolitis|peritonitis|endocarditis|otomastoiditis|pericarditis|pancolitis|mielitis|encefalitis
                                    |vasculitis|enterocolitis|colitis|proctitis|divertículitis|neumonitis|traqueobronquitis|bronquitis|orquiepididimitis|leptomeningitis
                                    |cerebritis|tiroiditis|meningoencefalitis|colecisititis)""", row['LECTURA'])][0]))

# for item in list2:
#     for x in item:
#         for y in x:
#             #print(y)
#             if y not in unique:
#                 unique.append(y)

#df['finding'] = np.array(list2)
df['count'] = np.array(list1)

#print(unique)
#lectura = df[df['LECTURA'].str.contains(pat = '(\bInfeccion|\bSepsis|\bSepticemia|\bNeumonia|\bChoque|Shock|\w*itis)', regex = True,na=False)]

df.to_excel(r'C:\Users\emmanuel.orrego\Documents\EIA\Tesis docs\Exceles\fichero.xlsx', index = False)