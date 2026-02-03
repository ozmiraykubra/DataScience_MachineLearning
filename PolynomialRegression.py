import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.pyplot import scatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\ozmir\Downloads\3-customersatisfaction.csv")
print(df.head())
df.drop("Unnamed: 0", axis=1, inplace=True)
print(df.head())
print(df.info())
plt.scatter(df["Customer Satisfaction"], df["Incentive"], color="blue")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
#plt.show()
plt.close()
# dependent & independent features
X = df[["Customer Satisfaction"]]
y = df["Incentive"]
print(X.head())
print(y.head())
# train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
print(X_train)
# scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regression = LinearRegression()
regression.fit(X_train, y_train)
# prediction
y_pred = regression.predict(X_test)
score = r2_score(y_test, y_pred)
print(score)
plt.scatter(X_train, y_train)
plt.plot(X_train, regression.predict(X_train), color = "r")
#plt.show()
plt.close()
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
print(X_test_poly)
regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)
score = r2_score(y_test, y_pred)
print(score)
print(regression.coef_)
print(regression.intercept_)
plt.scatter(X_train, y_train)
plt.scatter(X_train, regression.predict(X_train_poly), color = "r")
#plt.show()
poly = PolynomialFeatures(degree=3, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)
score = r2_score(y_test, y_pred)
print(score)
# new data
new_df = pd.read_csv(r"C:\Users\ozmir\Downloads\3-newdatas.csv")
print(new_df)
new_df.rename(columns={"0":"Customer Satisfaction"}, inplace=True)
print(new_df)
X_new = new_df[["Customer Satisfaction"]]
print(X_new)
X_new = scaler.fit_transform(X_new)
X_new_poly = poly.transform(X_new)
y_new = regression.predict(X_new_poly)
plt.plot(X_new, y_new, color = "red", label="New Prediction")
plt.scatter(X_train,y_train,label="Training Points")
plt.scatter(X_test,y_test,label="Test Points")
plt.legend()
#plt.show()
# pipeline
def poly_regression(degree):
    poly_features = PolynomialFeatures(degree=degree)
    ling_reg = LinearRegression()
    scaler = StandardScaler()
    pipeline =Pipeline([
        ("strandar_scaler", scaler),
        ("poly_features", poly_features),
        ("li-reg", ling_reg)
     ])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(score)
poly_regression(1)
