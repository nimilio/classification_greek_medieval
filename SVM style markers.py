import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('./features_new_train.csv', index_col=0)
df_test = pd.read_csv('./features_new_test.csv', index_col=0)

features_train = df_train.drop('Category',axis =1)
features_test = df_test.drop('Category',axis =1)

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


X_train = features_train
X_test = features_test
y_train = labels_train
y_test = labels_test

# ------------------------|
#  GRID SEARCH TUNING     |
# ------------------------|

param_grid = {
              'svc__C': [0.1,1,10,100,1000], 
              'svc__gamma':['auto',1,0.1,0.01,0.001,0.0001]}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy')
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))


# ------------------------|
#  MODEL FIT AND SCALER   |
# ------------------------|

from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
model = make_pipeline(StandardScaler(), svm.SVC(C = 10, kernel = 'rbf', gamma = 0.11))
model.fit(X_train, y_train)

predictions = model.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))




# ------------------------|
#  FEATURE INFORMATION    |
# ------------------------|

from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(model, X_test, y_test)
feature_names = X_train.columns
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")

for name, importance in zip(X_train.columns, perm_importance.importances_mean):
    print(name, "=", importance)




# ------------------------|
#      BEST FEATURES      |
# ------------------------|


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


best = SelectKBest(score_func=chi2, k=10)
fit = best.fit(X_train,y_train)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)


featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 
print(featureScores.nlargest(10,'Score'))