import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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


# -----------------------------|
#    HYPERPARAMETER TUNING     |
# -----------------------------|

from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state = 0)
param_grid = {'n_estimators': [100,200,300], 'n_jobs':[1,-1]}
grid = GridSearchCV(rf,param_grid,refit=True,verbose = 3)

grid.fit(X_train,y_train)


print(grid.best_params_)
print(grid.best_estimator_)


grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
print(grid.best_score_)
print(accuracy_score(y_test,grid_predictions))
# epigram, hymn, historical, poem, religious


# -----------------------------|
#         TRAIN MODEL          |
# -----------------------------|


rfc = RandomForestClassifier(n_estimators = 200, n_jobs = 1, random_state = 0)
rfc.fit(X_train,y_train)

for name, importance in zip(X_train.columns, rfc.feature_importances_):
    print(name, "=", importance)
print("\n")

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


# -----------------------------|
#      GINI COEFFICIENT        |
# -----------------------------|

#The higher, the more important the feature. The importance of a feature is computed as the (normalized) total 
#reduction of the criterion brought by that feature. It is also known as the Gini importance.
sorted_idx = rfc.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], rfc.feature_importances_[sorted_idx])

#The drawback of the method is a tendency to prefer (select as important) numerical features and categorical features 
#with high cardinality.


# -----------------------------|
#    PERMUTATION IMPORTANCE    |
# -----------------------------|


#measuring how score decreases when a feature is not available;
#The method is most suitable for computing feature importances when a number of columns (features) is not huge
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(rfc, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
for name, importance in zip(X_train.columns, perm_importance.importances_mean):
    print(name, "=", importance)
print("\n")
plt.barh(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")