from hmac import digest_size

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\ozmir\Downloads\4-Algerian_forest_fires_dataset.csv")
print(df.head())
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])
df.drop(122, inplace=True)
print(df[df.isnull().any(axis=1)])
df.loc[:123 , "Region"] = 0
df.loc[123: , "Region"] = 1
print(df.head())
print(df.tail())
print(df.isnull().any())
print(df.isnull().sum())
df = df.dropna().reset_index(drop=True)
print(df.isnull().sum())
print(df.shape)

df.columns = df.columns.str.strip()
print(df.columns)
print(df.info())
print(df["day"].unique())
print(df[df["day"] == "day"])
df.drop(122 , inplace=True)
print(df.iloc[122])
df[["day","month","year","Temperature","RH","Ws"]] = df[["day","month","year","Temperature","RH","Ws"]].astype(int)
print(df.info())
print(df.head())
df[["Rain", "FFMC", "DMC","DC","ISI","BUI","FWI"]] = df[["Rain", "FFMC", "DMC","DC","ISI","BUI","FWI"]].astype(float)
print(df.info())
print(df['Classes'].value_counts())
df['Classes'] = np.where(df['Classes'].str.contains('not fire'),0,1)
print(df['Classes'].value_counts())
print(df['Classes'].value_counts(normalize=True)*100)
sns.heatmap(df.corr())
#plt.show()
plt.close()
df.drop(['day','month','year'],axis=1,inplace=True)
print(df.head())

# dependent & independent features
X = df.drop("FWI",axis=1)
y = df["FWI"]
print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=15)
print(X_train.shape)
print(X_train.corr())

# redundancy , multicollinearity , overfitting
print(X_train.corr().columns)
def correlation_for_dropping(df , threshold):
    columns_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshold:
                columns_to_drop.add(corr.columns[i])
    return columns_to_drop

columns_dropping = correlation_for_dropping(X_train, 0.85)
X_train.drop(columns_dropping, axis=1, inplace = True)
X_test.drop(columns_dropping, axis=1, inplace = True)
print(X_train.shape)
print(X_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title('X_train')
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title('X_train_scaled')
#plt.show()
plt.close()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
elastic = ElasticNet()
elastic.fit(X_train_scaled, y_train)
y_pred = elastic.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", score)
plt.scatter(y_test, y_pred)
#plt.show()

# lasso cross validation
from sklearn.linear_model import LassoCV
lassocv = LassoCV(cv = 5)
lassocv.fit(X_train_scaled, y_train)
y_pred = lassocv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", score)
plt.scatter(y_test, y_pred)
#plt.show()

# ridge cross validation
from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(cv = 5)
ridgecv.fit(X_train_scaled, y_train)
y_pred = lassocv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", score)
plt.scatter(y_test, y_pred)
#plt.show()

plt.close()

#-------------------Regression Decesion-------------------------
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np

diabetes = datasets.load_diabetes()
print(type(diabetes))
print(diabetes.DESCR)
