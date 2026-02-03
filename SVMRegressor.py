import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from RidgeLassoElasticNet import X_train_scaled
from SVMClassifier import linear

df = pd.read_csv(r"C:\Users\ozmir\Downloads\10-diamonds.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.shape)
df = df.drop("Unnamed: 0", axis=1)
print(len(df[df["x"] == 0]), len(df[df["y"] == 0]),len(df[df["z"] == 0]) )
df = df.drop(df[df["x"] == 0].index)
df = df.drop(df[df["y"] == 0].index)
df = df.drop(df[df["z"] == 0].index)
print(df.describe())
print(df.shape)
sns.pairplot(df)
#plt.show()
plt.close()
sns.scatterplot(x=df["x"], y=df["price"])
#plt.show()
plt.close()
len(df[(df["depth"]<75)&(df["depth"]>45)])
len(df[(df["table"]<75)&(df["table"]>40)])
len(df[(df["z"]<30)&(df["z"]>2)])
len(df[(df["y"]<20)])
df = df[(df["depth"]<75)&(df["depth"]>45)]
df = df[(df["table"]<75)&(df["table"]>40)]
df = df[(df["z"]<30)&(df["z"]>2)]
df = df[(df["y"]<20)]
print(df.describe())
X = df.drop("price", axis=1)
y = df["price"]
print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in ["cut", "color", "clarity"]:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])
print(X_train.head())
print(X_train["cut"].value_counts())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_pred = linear.predict(X_test_scaled)
mae = mean_squared_error(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)
print("Mean absolute error: ", mae)
print("Mean squared error: ", mse)
print("R2 Score: ", r2)
plt.scatter(y_test, y_pred)
#plt.show()
plt.close()
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)
mae = mean_squared_error(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)
print("Mean absolute error: ", mae)
print("Mean squared error: ", mse)
print("R2 Score: ", r2)
plt.scatter(y_test, y_pred)
#plt.show()