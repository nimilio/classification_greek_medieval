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

data = df.drop('Category',axis =1).to_numpy()

pca = PCA(2,random_state = 0)
df2 = pca.fit_transform(data)
df2.shape


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


print(confusion_matrix(df['Cluster'],label))
target_names = ['Epigram','Hymn', 'Historical','Religious','Poetry',]
print(classification_report(df['Cluster'],label,target_names=target_names))


# --------------------------|
#    PREDICTED CLUSTERS     |
# --------------------------|

x = df.iloc[:, 1:16]
plot_labels = [target_names[i] for i in label]
sns.scatterplot(df2[:,0], df2[:,1], hue = plot_labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], marker='x', s=200, c='k')



# --------------------------|
#  GROUND TRUTH COMPARISON  |
# --------------------------|


a = df['Category'].tolist()
b = np.array(a)
sns.scatterplot(df2[:,0], df2[:,1], hue = b)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], marker='x', s=200, c='k')


# --------------------------|
#    HOMOGENEITY SCORES     |
# --------------------------|

homogeneity_score(df.Cluster, label)



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


