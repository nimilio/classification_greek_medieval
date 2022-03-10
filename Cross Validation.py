import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from pprint import pprint



# ---------------------|
# READ CSV AND PROCESS |
# ---------------------|


df = pd.read_csv('./features_new_all.csv', index_col=0)
features = df.drop('Category',axis =1)

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
    
df['Label'] = df['Category'].apply(converter)
labels = df['Label']

X = features
y = labels



# --------------------------|
# INITIATE CROSS VALIDATION |
# --------------------------|

accuracy = []
f1_scores = []
count = 1
folds = True

best_features = {}

accuacy1 = []
f1_scores1 = []

times = 0

rf = RandomForestClassifier(n_estimators = 200, n_jobs = 1, random_state = 0)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,random_state=0)


for train_index, test_index in cv.split(X,y):
    X1_train,X1_test = X.iloc[train_index], X.iloc[test_index]
    y1_train,y1_test = y.iloc[train_index], y.iloc[test_index]
    rf.fit(X1_train,y1_train)
    prediction = rf.predict(X1_test)
    
    score = accuracy_score(prediction,y1_test)
    accuracy.append(score)
    
    f1 = f1_score(prediction,y1_test, average='weighted')
    f1_scores.append(f1)

    for name, importance in zip(X.columns, rf.feature_importances_):
        if name in best_features:
            best_features[name] += importance
        elif name not in best_features:
            best_features[name] = importance
    times +=1


sorted_features = {k: v/times for k, v in sorted(best_features.items(), key=lambda item: item[1])}    
first_trial_accuracy = accuracy[0:5]
first_trial_F1 = f1_scores[0:5]




print(times) # for verification

print('\n')


# --------------------------|
#       PRINT RESULTS       |
# --------------------------|
print ("The accuracies of the first trial are: ",[i*100 for i in first_trial_accuracy])
print ("The F1 scores of the first trial are: ",[i*100 for i in first_trial_F1])

print('\n')

print ("The mean accuracy of the first trial is: ",np.array(first_trial_accuracy).mean()*100)
print ("and its standard deviation is: ",np.array(first_trial_accuracy).std()*100)

print('\n')

print ("The mean F1 score of the first trial is: ",np.array(first_trial_F1).mean()*100)
print ("and its standard deviation is: ",np.array(first_trial_F1).std()*100)

print('\n')

print ("The overall mean accuracy is: ",np.array(accuracy).mean()*100)
print ("and its standard deviation is: ",np.array(accuracy).std()*100)

print('\n')

print ("The overall mean F1 score is: ",np.array(f1_scores).mean()*100)
print ("and its standard deviation is: ",np.array(f1_scores).std()*100)
