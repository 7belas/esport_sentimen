import pandas as pd
import numpy as np
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# PERSIAPAN DATA
data = pd.read_excel('data/data_tweet.xlsx')
df = pd.DataFrame(data)

# CASE FOLDING
def case_folding(text) :
    return text.lower()
df['Case Folding'] = df['Tweet'].apply(case_folding)
df['Tokenizing'] = df['Tweet'].apply(case_folding)

# TOKENIZING
def remove_tweet_special(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    return text.replace("http://", " ").replace("https://", " ")
df['Tokenizing'] = df['Tokenizing'].apply(remove_tweet_special)

def remove_number(text):
    return  re.sub(r"\d+", "", text)
df['Tokenizing'] = df['Tokenizing'].apply(remove_number)

def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))
df['Tokenizing'] = df['Tokenizing'].apply(remove_punctuation)

def remove_whitespace_LT(text):
    return text.strip()
df['Tokenizing'] = df['Tokenizing'].apply(remove_whitespace_LT)

def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)
df['Tokenizing'] = df['Tokenizing'].apply(remove_whitespace_multiple)

def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)
df['Tokenizing'] = df['Tokenizing'].apply(remove_singl_char)

def word_tokenize_wrapper(text):
    return word_tokenize(text)
df['Tokenizing'] = df['Tokenizing'].apply(word_tokenize_wrapper)

# FILTERING (STOPWORD REMOVAL)
list_stopwords = stopwords.words('indonesian')

list_stopwords = set(list_stopwords)

def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

df['Stopword Removal'] = df['Tokenizing'].apply(stopwords_removal)

# OUTPUT KE EXCEL 
df.to_excel('data/data_preprocesing.xlsx', columns=["Tweet", "Case Folding", "Tokenizing", "Stopword Removal"], index=False, header=True)
