import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Load data dari file Excel
data = pd.read_excel('data/data_analisis2.xlsx')

# Filter data dengan sentimen positif
positive_words = data.loc[data['Labeling'] == 'POSITIF']['Tweet'].str.cat(sep=' ')

# Filter data dengan sentimen negatif
negative_words = data.loc[data['Labeling'] == 'NEGATIF']['Tweet'].str.cat(sep=' ')

# Menghitung kemunculan kata pada sentimen positif
positive_word_count = Counter(positive_words.split())

# Menghitung kemunculan kata pada sentimen negatif
negative_word_count = Counter(negative_words.split())

# Mengambil 20 kata yang paling sering muncul pada sentimen positif
top_positive_words = dict(positive_word_count.most_common(20))

# Mengambil 20 kata yang paling sering muncul pada sentimen negatif
top_negative_words = dict(negative_word_count.most_common(20))

# Membuat object WordCloud untuk sentimen positif
wordcloud_positive = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=[],
                              min_font_size=10).generate(positive_words)

# Membuat object WordCloud untuk sentimen negatif
wordcloud_negative = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=[],
                              min_font_size=10).generate(negative_words)

# Plot WordCloud untuk sentimen positif
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_positive)
plt.axis("off")
plt.tight_layout(pad=0)
plt.title("Wordcloud Sentimen Positif")
plt.show()

# Plot WordCloud untuk sentimen negatif
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_negative)
plt.axis("off")
plt.tight_layout(pad=0)
plt.title("Wordcloud Sentimen Negatif")
plt.show()

# Menampilkan grafik kemunculan kata pada sentimen positif
plt.figure(figsize=(10, 5))
plt.bar(top_positive_words.keys(), top_positive_words.values())
plt.xticks(rotation='vertical')
plt.xlabel('Kata')
plt.ylabel('Jumlah Kemunculan')
plt.title('Kemunculan 20 Kata pada Sentimen Positif')

# Menampilkan grafik kemunculan kata pada sentimen negatif
plt.figure(figsize=(10, 5))
plt.bar(top_negative_words.keys(), top_negative_words.values())
plt.xticks(rotation='vertical')
plt.xlabel('Kata')
plt.ylabel('Jumlah Kemunculan')
plt.title('Kemunculan 20 Kata pada Sentimen Negatif')

# Menampilkan grafik
plt.show()
