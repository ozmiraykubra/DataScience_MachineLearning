import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from seaborn.matrix import dendrogram

df = pd.read_csv(r"C:\Users\ozmir\Downloads\27-mall_customers.csv")
print(df.head())
print(df.info())
print(df.describe())

import math
def plot_all_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
#    plt.show()
plot_all_histograms(df)
print(df['Gender'].value_counts())
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
print(df.head())
df = df.drop('CustomerID', axis=1)
print(df.head())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns = df.columns)
print(df.head())
plot_all_histograms(df)

import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10, 8))
dendrogram = sch.dendrogram(sch.linkage(df, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customer")
plt.ylabel("Distance")
#plt.show()

# we can clearly select 4 to 6 clusters from this dendrogram, let's go for 4
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4)
hc.fit_predict(df)
y_hc = hc.fit_predict(df)
print(y_hc)
df['cluster'] = pd.DataFrame(y_hc)
print(df.head())
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="cluster")
plt.title("Customer Clusters")
#plt.show()
from sklearn.metrics import  silhouette_score
print(silhouette_score(df, y_hc))

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()
print(X.head())

hc = AgglomerativeClustering(n_clusters=4)
y_hc = hc.fit_predict(X)
X['cluster'] = y_hc
sns.scatterplot(data=X, x="Annual Income (k$)", y="Spending Score (1-100)", hue="cluster")
plt.title("Customer Clusters")
#plt.show()
print(silhouette_score(X, y_hc))

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
df = pd.read_csv(r"C:\Users\ozmir\Downloads\27-mall_customers.csv")
df = df.drop('CustomerID', axis=1)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

features_2d = ["Annual Income (k$)", "Spending Score (1-100)"]
features_3d = ["Age" ,"Annual Income (k$)", "Spending Score (1-100)"]
features_4d = ["Gender","Age" ,"Annual Income (k$)", "Spending Score (1-100)"]

for feats in [features_2d, features_3d, features_4d]:
    X = df[feats]
    X_scaled = MinMaxScaler().fit_transform(X)

    hc = AgglomerativeClustering(n_clusters=5)
    y_hc = hc.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, y_hc)
    db = davies_bouldin_score(X_scaled, y_hc)
    ch = calinski_harabasz_score(X_scaled, y_hc)

    print("features: {feats}")
    print("Silhoutte score: ",sil)
    print("Davis Bouldin score: ",db)
    print("Calinski Harabasz score: ",ch)
    print("----------------------------")

from sklearn.cluster import KMeans
df = pd.read_csv(r"C:\Users\ozmir\Downloads\27-mall_customers.csv")
df = df.drop('CustomerID', axis=1)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

features_2d = ["Annual Income (k$)", "Spending Score (1-100)"]
features_3d = ["Age" ,"Annual Income (k$)", "Spending Score (1-100)"]
features_4d = ["Gender","Age" ,"Annual Income (k$)", "Spending Score (1-100)"]

for feats in [features_2d, features_3d, features_4d]:
    X = df[feats]
    X_scaled = MinMaxScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=5)
    y_hc = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, y_hc)
    db = davies_bouldin_score(X_scaled, y_hc)
    ch = calinski_harabasz_score(X_scaled, y_hc)

    print("features: {feats}")
    print("Silhoutte score: ",sil)
    print("Davis Bouldin score: ",db)
    print("Calinski Harabasz score: ",ch)
    print("----------------------------")

df = pd.read_csv(r"C:\Users\ozmir\Downloads\27-mall_customers.csv")
df = df.drop('CustomerID', axis=1)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

X = df[["Annual Income (k$)", "Spending Score (1-100)"]].copy()

X_scaled = MinMaxScaler().fit_transform(X)
hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(X_scaled)
X_scaled = pd.DataFrame(X_scaled)
X_scaled['cluster'] = y_hc
sil = silhouette_score(X_scaled, y_hc)
print(sil)