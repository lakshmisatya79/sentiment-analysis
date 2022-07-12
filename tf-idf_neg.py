from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import io
import json
import os
import csv
import pandas as pd
df = pd.read_csv('neg_train.csv')
s = df.text
#with io.open('positive_train.txt', 'r') as txtfile:
# Lines = txtfile.readlines()

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(line for line in s)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()

df1= pd.DataFrame(denselist, columns=feature_names)
#print(df1.head)
with io.open('neg_words_score.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['text', 'score'])
    for (line,d) in zip(s,denselist):
       writer.writerow([line,-sum(d)])

# default CSV
#df1.to_csv('pos_words_score.csv',header = ['score'] , index= False)
#print('\nCSV String:\n', csv_data)
