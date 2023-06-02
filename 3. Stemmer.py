import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from ast import literal_eval

# PERSIAPAN DATA
data = pd.read_excel('data/data_preprocesing.xlsx', usecols=["Tweet", "Stopword Removal"])
df = pd.DataFrame(data)
df['Stopword Removal'] = df['Stopword Removal'].apply(literal_eval)

# STEMMER
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmed_wrapper(text):
    return stemmer.stem(text)

term_dict = {}
for document in df['Stopword Removal'] :
    for term in document :
        if term not in term_dict :
            term_dict[term] = ' '

for term in term_dict :
    term_dict[term] = stemmed_wrapper(term)

def get_stemmed_term(document):
    return [term_dict[term] for term in document]

df['Stemmer'] = df['Stopword Removal'].swifter.apply(get_stemmed_term)

df.to_excel('data/data_preprocesing2.xlsx', index=False, header=True)
