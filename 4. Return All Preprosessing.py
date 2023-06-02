import pandas as pd
import numpy as np
from ast import literal_eval

# PERSIAPAN DATA
data = pd.read_excel('data/data_preprocesing2.xlsx')
df = pd.DataFrame(data)

df['Stemmer'] = df['Stemmer'].apply(literal_eval)

# MENGEMBALIKAN DATA HASIL TOKENIZING DAN REMOVAL STOPWORD
def fit_stopword(text) :
    text = np.array(text)
    text = ' '.join(text)
    return text
df['Preprocesing'] = df['Stemmer'].apply(lambda x: fit_stopword(x))

# DROP DATA YANG KOSONG / EMPTY STRING
df['Preprocesing'].replace('', np.nan, inplace=True) 
df.dropna(inplace=True)

df.to_excel('data/data_preprocesing3.xlsx', index=False, header=True) 