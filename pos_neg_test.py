import pandas as pd
import csv
import io

df1=pd.read_csv('pos_words_score_test.csv')
s=df1
df2=pd.read_csv('neg_words_score_test.csv')
s1=df2
s3=pd.concat([s,s1])
s3.to_csv (r'final_test.csv',header=['text','score'], index=None)
#print(s3.text)