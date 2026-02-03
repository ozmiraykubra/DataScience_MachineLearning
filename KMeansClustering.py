import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans

df = pd.read_csv(r"C:\Users\ozmir\Downloads\26-customer_data.csv")
print(df.head())
print(df.info())
sns.scatterplot(data=df, x="Annual_Income", y="Spending_Score")
#plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.2, random_state=15)
print(X_train)

from sklearn.preprocessing import minmax_scale, MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.cluster import KMeans

#elbow method
wcss =[]
for k in range (1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)
plt.plot(range(1,11), wcss)
plt.xticks(range(1,11))
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
#plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train_scaled)
y_pred = kmeans.predict(X_test_scaled)
sns.scatterplot(data=pd.DataFrame(X_test_scaled, columns=X_test.columns), x="Annual_Income", y="Spending_Score", hue =y_pred)
#plt.show()

from kneed import KneeLocator
kl = KneeLocator(range(1,11), wcss, curve="convex", direction="decreasing")
print(kl.elbow)

# silhoutte score
from sklearn.metrics import silhouette_score
silhouette_scores =[]
for k in range (2,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X_train_scaled)
    score = silhouette_score(X_train_scaled, kmeans.labels_)
    silhouette_scores.append(score)
plt.plot(range(2,11), silhouette_scores)
plt.xticks(range(2,11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette coefficient")
#plt.show()


