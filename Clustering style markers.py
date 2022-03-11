import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import homogeneity_score


df = pd.read_csv('./features_new_all.csv', index_col=0)



# ---------------|
#  ELBOW METHOD  |
# ---------------|


wcss = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(df.drop('Category',axis =1))
    wcss.append(km.inertia_)


plt.figure(figsize=(6, 6))
plt.plot(list_k, wss, '-o')
plt.xlabel(r'Number of clusters')
plt.ylabel('Within Cluster Sum of Square');

centr = pd.DataFrame({'Clusters':list_k,'WCSS': wcss})
print(centr)


# ---------------------|
#      MODEL FIT       |
# ---------------------|

kmeans = KMeans(5)
kmeans.fit(df.drop('Category',axis =1))
print(kmeans.cluster_centers_)



def converter(category):
    if category == 'Epigram':
        return 0
    elif category == 'Hymn':
        return 1
    elif category == 'Historical':
        return 2
    elif category == 'Religious':
        return 3
    elif category == 'Poetry':
        return 4
    
df['Cluster'] = df['Category'].apply(converter)

print(confusion_matrix(df['Cluster'],kmeans.labels_))
target_names = ['Epigram','Hymn', 'Historical','Religious','Poetry',]
print(classification_report(df['Cluster'],kmeans.labels_,target_names=target_names))




# --------------------------|
#    PCA FOR VISUALIZATION  |
# --------------------------|


# reduce the features to 2D
pca = PCA(n_components=2,random_state = 0)
principalComponents = pca.fit_transform(df.drop(['Category', 'Cluster'],axis =1))


# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)


# --------------------------|
#    PREDICTED CLUSTERS     |
# --------------------------|

plot_labels = [target_names[i] for i in kmeans.predict(df.drop(['Category', 'Cluster'],axis =1))]
sns.scatterplot(principalComponents[:,0], principalComponents[:,1], hue = plot_labels)
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=100, c='k')



# --------------------------|
#  GROUND TRUTH COMPARISON  |
# --------------------------|
a = df['Category'].tolist()
b = np.array(a)
sns.scatterplot(principalComponents[:,0], principalComponents[:,1], hue = b)
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=100, c='k')



# --------------------------|
#    HOMOGENEITY SCORES     |
# --------------------------|
 
X = df.iloc[:, 1:16].values  #from second column and ahead
homogeneity_score(df.Cluster, kmeans.labels_)




