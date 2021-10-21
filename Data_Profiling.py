import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_excel('C:\\Users\\emmanuel.orrego\\Documents\\EIA\\Tesis docs\\Exceles\\automationV2.xlsx')

prof = ProfileReport(df)
prof.to_file(output_file='output.html')