import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_files

df_train = pd.read_csv('./features_new_train.csv', index_col=0)
df_test = pd.read_csv('./features_new_test.csv', index_col=0)



def converter(category):
    if category == 'Epigram':
        return 0
    elif category == 'Hymn':
        return 1
    elif category == 'Historical':
        return 2
    elif category == 'Poetry':
        return 3
    elif category == 'Religious':
        return 4
df_train['Label'] = df_train['Category'].apply(converter)
df_test['Label'] = df_test['Category'].apply(converter)

labels_train = df_train['Label']
labels_test = df_test['Label']


df_train = df_train.reset_index()
df_train['Text'].iloc[0]
df_train['Text'].iloc[-1]
df_train.head()

df_test = df_test.reset_index()
df_test['Text'].iloc[0]
df_test['Text'].iloc[-1]
df_test.head()


# -------------------|
#   TF-IDF FEATURES  |
# -------------------|


tfidf_vectorizer = TfidfVectorizer()
vec_train = tfidf_vectorizer.fit(df_train['Text'])
vec_test = tfidf_vectorizer.fit(df_test['Text'])
features_train = tfidf_vectorizer.transform(df_train['Text'])
features_test = tfidf_vectorizer.transform(df_test['Text'])
print (features_train.shape,features_test.shape) # for verification



X_train = features_train
X_test = features_test
y_train = labels_train
y_test = labels_test


# --------------------|
#  FIT FEATURES TO RF |
# --------------------|

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, n_jobs = 1, random_state = 0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))