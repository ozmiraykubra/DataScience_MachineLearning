import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from SVMRegressor import y_pred

df = pd.read_csv(r"C:\Users\ozmir\Downloads\12-health_risk_classification.csv")
print(df.head())
print(df.info())
print(df.describe())
sns.scatterplot(x=df["blood_pressure_variation"], y=df["activity_level_index"], hue=df["high_risk_flag"])
#plt.show()
print(df["high_risk_flag"].value_counts())
X = df.drop("high_risk_flag", axis=1)
y = df["high_risk_flag"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
sns.boxplot(df)
#plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree",weights="uniform")
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("accuracy score: \n", accuracy_score(y_test, y_pred))
print("classification report: \n", classification_report(y_test, y_pred))
classifier = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree",weights="uniform")
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)
print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("accuracy score: \n", accuracy_score(y_test, y_pred))
print("classification report: \n", classification_report(y_test, y_pred))

classifier = KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree",weights="uniform")
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)
print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("accuracy score: \n", accuracy_score(y_test, y_pred))
print("classification report: \n", classification_report(y_test, y_pred))

df_reg = pd.read_csv(r"C:\Users\ozmir\Downloads\12-house_energy_regression.csv")
print(df_reg.info())
print(df_reg.head())
print(df_reg.describe())

