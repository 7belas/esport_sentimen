import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

data = pd.read_excel('data/data_analisis2.xlsx')
df1 = pd.DataFrame(data)


tfidf_vectorizer = TfidfVectorizer()
doc_vec = tfidf_vectorizer.fit_transform(data['Preprocesing'])

countvectorizer = CountVectorizer()
doc_b = countvectorizer.fit_transform(data['Preprocesing'])



df2 = pd.DataFrame(doc_vec.toarray().transpose(),
                   index=tfidf_vectorizer.get_feature_names_out(),)
df3 = pd.DataFrame(doc_b.toarray().transpose(),
                   index=countvectorizer.get_feature_names_out())

df2.to_excel('data/TF-IDF.xlsx') 
df3.to_excel('data/DF.xlsx') 
