import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.ensemble import RandomForestClassifier as RF

#  Importing the dataset
dataset = pd.read_csv("final_train1.csv")
x_train= dataset.text
y_train= dataset.sentiment

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=1)
dataset1 = pd.read_csv("final_test1.csv")
x_test1 = dataset1.text
y_test1 = dataset1.sentiment

#feature extraction
#cv = CountVectorizer()
#x_train_df = cv.fit_transform(x_train)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(x_train)
#print("dimensions of training data",x_train_df.shape)

#tf-idf transformer
#tfidf_transformer=TfidfTransformer()
#x_train_tf_idf=tfidf_transformer.fit_transform(x_train_df)

#using calssifier
dt = RF().fit(vectors, y_train)
#x_test_df = cv.fit_transform(x_test)
#x_test_df=cv.transform(x_test)
#x_test2=tfidf_transformer.transform(x_test_df)
x_test2=vectorizer.transform(x_test1)

# Predicting a new result
y_test=dt.predict(x_test2)

#df = pd.DataFrame(y_test1)
#df.to_csv (r'decision_trees.csv',header=['score'], index=None)


#printing resuts
from sklearn.metrics import recall_score, accuracy_score,f1_score

print ('Accuracy:', accuracy_score(y_test, y_test1))
print ('F1 score:', f1_score(y_test, y_test1,average='micro'))
print ('Recall:', recall_score(y_test, y_test1,average='micro'))

"""
# 6 Visualising  results
#plt.plot(accuracy_score(y_test2, y_test1), , color = 'green')
plt.title('random_forest Model')
plt.xlabel('predicted vs actual values')
# plt.ylabel('')
# plt.show()

plt.plot( [x1 for x1 in y_test],label='Predicted value')
plt.plot([x1 for x1 in y_test1],label='Actual value')
#plt.plot(y,x_test1)
#plt.plot(x, [xi*3 for xi in x])
plt.legend()
plt.show()
"""