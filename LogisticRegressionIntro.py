import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\ozmir\Downloads\6-bank_customers.csv")
print(df.columns)
print(df.head())
print(df.describe())
X = df.drop("subscribed",axis=1)
y = df["subscribed"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=15)
from sklearn.linear_model import LinearRegression, LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
score = accuracy_score(y_test, y_pred)
print("score: ",score)
print(classification_report(y_pred,y_test))
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))

# hyperparameter tuning
model = LogisticRegression()
penalty = ["l1", "l2" , "elasticnet"]
c_values = [100, 10 , 1 ,0.1, 0.01]
solver = ["newton-cg", "liblinear", "sag", "saga", "newton-cholesky"]

params = dict(penalty=penalty, C=c_values, solver=solver)
print(params)

# grid search cv
from sklearn.model_selection import GridSearchCV, StratifiedKFold
cv = StratifiedKFold()
grid = GridSearchCV(estimator=model, param_grid= params, cv=cv, scoring="accuracy",n_jobs=-1)
print(grid.fit(X_train, y_train))
print(grid.best_params_)
print(grid.best_score_)
y_pred = grid.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("score: ",score)
print(classification_report(y_pred,y_test))
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))
# random search cv
from sklearn.model_selection import RandomizedSearchCV
model = LogisticRegression()
randomcv = RandomizedSearchCV(estimator = model , param_distributions=params, cv=5, n_iter=10 , scoring="accuracy")
randomcv.fit(X_train, y_train)
print(randomcv.best_params_)
print(randomcv.best_score_)
Y_pred = randomcv.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("score: ",score)
print(classification_report(y_pred,y_test))
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))

#---------------LogisticRegressionMultiClass---------------
df = pd.read_csv(r"C:\Users\ozmir\Downloads\7-cyber_attack_data.csv")
print(df.columns)
print(df.head())
print(df.info())
X = df.drop("attack_type",axis=1)
y = df["attack_type"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=15)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix, confusion_matrix
score  = accuracy_score(y_pred, y_test)
print("score: ",score)
print(classification_report(y_pred, y_test))
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))
penalty = ["l1", "l2" , "elasticnet"]
c_values = [100, 10 , 1 ,0.1, 0.01]
solver = ["newton-cg","lbfgs" ,"liblinear","sag","saga","newton-cholesky"]
params = dict(penalty=penalty, C=c_values, solver=solver)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
cv = StratifiedKFold()
grid = GridSearchCV(estimator= model, param_grid= params, cv=cv, scoring="accuracy",n_jobs=-1)
print(grid.fit(X_train, y_train))
print(grid.best_params_)
print(grid.best_score_)
y_pred = grid.predict(X_test)
score = accuracy_score(y_pred, y_test )
print("score: ",score)
print(classification_report(y_pred, y_test))
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))

# one vs rest
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
onevsonemodel = OneVsOneClassifier(LogisticRegression())
onevsrestmodel = OneVsRestClassifier(LogisticRegression())
onevsonemodel.fit(X_train, y_train)
y_pred = onevsonemodel.predict(X_test)
score = accuracy_score(y_pred, y_test )
print("score: ",score)
print(classification_report(y_pred, y_test))
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))

onevsrestmodel.fit(X_train, y_train)
y_pred = onevsrestmodel.predict(X_test)
score = accuracy_score(y_pred, y_test )
print("score: ",score)
print(classification_report(y_pred, y_test))
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))

#----------LogisticRegressionAdvanced-------------
df = pd.read_csv(r"C:\Users\ozmir\Downloads\7-cyber_attack_data.csv")
print(df.head())
print(df['is_fraud'].value_counts())

