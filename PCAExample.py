import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from XGBoostRegressor import columns

data = load_breast_cancer(as_frame=True)
df = data.frame
print(df.head())
print(df.shape)
print(data.DESCR)
print(data.feature_names)
print(df.info())
print(df.isnull().sum())
X = df.drop('target', axis=1)
y = df['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=15)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
columns = data.feature_names
X_train = pd.DataFrame(X_train, columns=columns)
X_test = pd.DataFrame(X_test, columns=columns)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
logistic = LogisticRegression()
gbc = GradientBoostingClassifier()
logistic.fit(X_train, y_train)
gbc.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("logistic regression")
y_pred = logistic.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("---------------------------------")

print("Gradient Boosting Classifier")
y_pred_gbc = gbc.predict(X_test)
print(accuracy_score(y_test, y_pred_gbc))
print(confusion_matrix(y_test, y_pred_gbc))
print(classification_report(y_test, y_pred_gbc))

from sklearn.decomposition import  PCA
pca = PCA(n_components=2)
pca.fit_transform(X_train)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca)

X_train_pca = pd.DataFrame(X_train_pca, columns=["PC 1", "PC 2"])
X_test_pca = pd.DataFrame(X_test_pca, columns=["PC 1", "PC 2"])
print(X_train_pca)

logreg = LogisticRegression()
gbc = GradientBoostingClassifier()

logistic.fit(X_train_pca, y_train)
gbc.fit(X_train_pca, y_train)

print("logistic regression")
y_pred = logistic.predict(X_test_pca)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("---------------------------------")

print("Gradient Boosting Classifier")
y_pred_gbc = gbc.predict(X_test_pca)
print(accuracy_score(y_test, y_pred_gbc))
print(confusion_matrix(y_test, y_pred_gbc))
print(classification_report(y_test, y_pred_gbc))

X = df.drop("target", axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=2)
X = pca.fit_transform(X)

X = pd.DataFrame(X, columns=["PC1","PC2"])
sns.scatterplot(data = X, x="PC1", y="PC2", hue=df["target"])
#plt.show()

