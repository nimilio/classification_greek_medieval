import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.datasets import load_files
import seaborn as sns


df = pd.read_csv('./features_new_all.csv', index_col=0)



# --------------------------|
#      ELBOW METHOD         |
# --------------------------|

wcss = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(features)
    wcss.append(km.inertia_)


plt.figure(figsize=(6, 6))
plt.plot(list_k, wcss, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Within Cluster Sum of Square');

centr = pd.DataFrame({'Clusters':list_k,'WCSS': wcss})
centr


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
    
df['Label'] = df['Category'].apply(converter)

labels = df['Label']



df = df.reset_index()
df['Text'].iloc[0]
df['Text'].iloc[-1]
print(df.head())



tfidf_vectorizer = TfidfVectorizer()
vec = tfidf_vectorizer.fit(df['Text'])
features = tfidf_vectorizer.transform(df['Text'])
print(features.shape)



# ---------------------|
#      MODEL FIT       |
# ---------------------|

cls = KMeans(n_clusters = 5)
kmeans_centers = cls.fit(features)
kmeans_labels = cls.predict(features)
print(kmeans_labels)
print(kmeans_centers.cluster_centers_)


print(confusion_matrix(df['Label'],cls.labels_))
target_names = ['Epigram','Hymn', 'Historical','Religious','Poetry',]
print(classification_report(df['Label'],cls.labels_,target_names=target_names))




# --------------------------|
#    PCA FOR VISUALIZATION  |
# --------------------------|


# reduce the features to 2D
pca = PCA(n_components=2, random_state=0)
principalComponents = pca.fit_transform(features.toarray())

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)



# --------------------------|
#    PREDICTED CLUSTERS     |
# --------------------------|

plt.scatter(principalComponents[:,0], principalComponents[:,1], c=cls.predict(features))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


# --------------------------|
#  GROUND TRUTH COMPARISON  |
# --------------------------|

plt.scatter(principalComponents[:,0], principalComponents[:,1], c=df.Label.values)
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


# --------------------------|
#    HOMOGENEITY SCORES     |
# --------------------------|

from sklearn.metrics import homogeneity_score
homogeneity_score(df.Label, cls.predict(features))






# --------------------------|
#    PCA BEFORE CLUSTERING  |
# --------------------------|


pca2 = PCA(2,random_state = 0)
df2 = pca2.fit_transform(features.toarray())
print(df2.shape)


# --------------------------|
#      ELBOW METHOD         |
# --------------------------|

wsse = []
list_k2 = list(range(1, 10))

for k in list_k2:
    km = KMeans(n_clusters=k)
    km.fit(df2)
    wsse.append(km.inertia_)


plt.figure(figsize=(6, 6))
plt.plot(list_k2, wsse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

centr2 = pd.DataFrame({'Clusters':list_k2,'WSS': wss})
print(centr2)



# ---------------------|
#      MODEL FIT       |
# ---------------------|

kmeans2 = KMeans(n_clusters= 5)
label2 = kmeans2.fit_predict(df2)
kmeans_labels = kmeans2.predict(df2)
print(label2)


print(confusion_matrix(df['Label'],kmeans2.labels_))
target_names = ['Epigram','Hymn', 'Historical','Religious','Poetry']
print(classification_report(df['Label'],kmeans2.labels_,target_names=target_names))






# --------------------------|
#    PREDICTED CLUSTERS     |
# --------------------------|
 
reduced_cluster_centers2 = kmeans2.cluster_centers_

plot_labels = [target_names[i] for i in kmeans2.predict(df2)]
sns.scatterplot(df2[:,0], df2[:,1], hue = plot_labels)
plt.scatter(reduced_cluster_centers2[:, 0], reduced_cluster_centers2[:,1], marker='x', s=200, c='k')



# --------------------------|
#  GROUND TRUTH COMPARISON  |
# --------------------------|


a = df['Category'].tolist()
b = np.array(a)
sns.scatterplot(df2[:,0], df2[:,1], hue = b)
plt.scatter(reduced_cluster_centers2[:, 0], reduced_cluster_centers2[:,1], marker='x', s=200, c='k')


# --------------------------|
#    HOMOGENEITY SCORES     |
# --------------------------|


from sklearn.metrics import homogeneity_score
homogeneity_score(df.Label, kmeans2.labels_)