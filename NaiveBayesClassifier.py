import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\ozmir\Downloads\11-iris.csv")
print(df.head())
print(df.isnull().sum())
print(df["Species"].value_counts())
print(df.describe())
sns.pairplot(df)
#plt.show()
plt.close()

print(df.columns)
sns.scatterplot(x=df["SepalLengthCm"], y=df["SepalWidthCm"], hue=df["Species"])
#plt.show()
plt.close()

print(df.head())
df = df.drop("Id",axis=1)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])
print(df.head())
print(df.tail())
print(df["Species"].value_counts())
X = df.drop("Species",axis=1)
y = df["Species"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_ped = gnb.predict(X_test_scaled)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("confusion matrix: \n", confusion_matrix(y_test, y_ped))
print("accuracy score: \n", accuracy_score(y_test, y_ped))
print("classification report: \n", classification_report(y_test, y_ped))
