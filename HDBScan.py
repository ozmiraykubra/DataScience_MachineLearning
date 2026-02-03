from cProfile import label

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r"C:\Users\ozmir\Downloads\28-urban_pedestrian_locations_with_labels.csv")
print(df.head())
print(df.info())

sns.scatterplot(data=df , x="x_position", y="y_position")
#plt.show()

df = df.drop("true_cluster", axis=1)
print(df.head())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
from sklearn.cluster import  DBSCAN
dbscan = DBSCAN()
dbscan.fit(X_scaled)
print(dbscan.labels_)
X_scaled = pd.DataFrame(X_scaled, columns=["x_position", "y_position"])
sns.scatterplot(data=X_scaled, x="x_position", y="y_position", hue= dbscan.labels_)
#plt.show()

df = pd.read_csv(r"C:\Users\ozmir\Downloads\28-urban_pedestrian_locations_with_labels.csv")
print(df.columns)
sns.scatterplot(data=df, x="x_position", y="y_position", hue="true_cluster")
#plt.show()

eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
min_samples_values = [4,5,6]

from sklearn.metrics import silhouette_score
results = []
for eps in eps_values:
    for min_samples in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_

        if len(set(labels)) <= 1:
            continue

        silhouette = silhouette_score(X_scaled, labels)
        results.append(
            {
                "eps": eps,
                "min_samples": min_samples,
                "Silhoutte score": silhouette,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            }
        )
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Silhoutte score", ascending=False)
print(results_df)

from sklearn.cluster import HDBSCAN
hdbscan = HDBSCAN()
hdbscan.fit(X_scaled)
print(hdbscan.labels_)
if __name__ == "__main__":
    import pandas as pd
    X_scaled_df = pd.DataFrame(X_scaled, columns=["x_position", "y_position"])
    X_scaled_df["labels_"] = labels

    sns.scatterplot(data=X_scaled_df, x="x_position", y="y_position", hue="labels_")
    plt.title("HDBScan Kümeleme Görselleştirmesi")
    #plt.show()

min_cluster_size = [3,5,7,10]
min_samples = [None, 3,5,7]
results = []
for min_cluster in min_cluster_size:
    for min_sample in min_samples:
        hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_sample).fit(X_scaled)
        labels = hdb.labels_

        if len(set(labels)) <= 1:
            continue

        silhouette = silhouette_score(X_scaled, labels)
        results.append(
            {
                "min_cluster_size": min_cluster,
                "min_samples": min_sample,
                "Silhoutte score": silhouette,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            }
        )
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Silhoutte score", ascending=False)
print(results_df)

