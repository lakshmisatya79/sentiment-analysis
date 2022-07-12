import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import csv

import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR

# 2 Importing the dataset
dataset = pd.read_csv("final_train1.csv")
X = dataset.text
y = dataset.Score

dataset1 = pd.read_csv("final_test.csv")
x_test1 = dataset1.text


# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
# print(x_test)

cv = CountVectorizer(max_features=100000, stop_words='english', ngram_range=(1, 3), max_df=0.2)
x_train_df = cv.fit_transform(x_train)
x_test_df = cv.transform(x_test)
x_test2 = cv.transform(x_test1)
# X_test1_df = cv.transform(x_test1)

rfc = SVR()
rfc.fit(x_train_df, y_train)
# print(rfc.score(x_test_df, y_test))
print(rfc.score(x_test_df, y_test))
print(y_test)

# print(rfc.score(X_test2, y_test))
# print(rfc.predict(X_test2))
print(x_test1)
#print(x_test2)
y_test1=rfc.predict((x_test2))
print(y_test1)
df = pd.DataFrame(y_test1)
df.to_csv (r'softmax.csv',header=['score'], index=None)
# 5 Predicting a new result
from sklearn.metrics import recall_score, accuracy_score,f1_score

print ('Accuracy:', accuracy_score(y_test2, y_test1))
print ('F1 score:', f1_score(y_test2, y_test1,average='micro'))
print ('Recall:', recall_score(y_test2, y_test1,average='micro'))
# 6 Visualising the Support Vector Regression results
##plt.plot(X, regressor.predict(X), color = 'green')
# plt.title('Truth or Bluff (Support Vector Regression Model)')
# plt.xlabel('')
# plt.ylabel('Salary')
# plt.show()
