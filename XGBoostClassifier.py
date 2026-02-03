import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyarrow.interchange.from_dataframe import column_to_array

from LogisticRegressionIntro import params, y_pred

df = pd.read_csv(r"C:\Users\ozmir\Downloads\20-digitalskysurvey.csv")
print(df.head())
column_to_drop = ["objid","specobjid","rerun","camcol","field","run"]
print(df.drop(columns = column_to_drop,axis = 1, inplace=True))
print(df['class'].value_counts())
sns.scatterplot(data=df, x="redshift", y="ra", hue="class")
#plt.show()
sns.scatterplot(data=df, x="redshift", y="plate", hue="class")
#plt.show()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])
print(df.head())
print(df.corr())
sns.countplot(data=df, x="mjd", hue="class")
#plt.show()
sns.scatterplot(data=df, x="mjd", y="redshift", hue="class")
#plt.show()
sns.pairplot(df, hue="class")
#plt.show()
fig, axis = plt.subplots(nrows=1,ncols=3,figsize=(16,4))
ax = sns.histplot(df[df["class"] == 2],y="redshift", ax=axis[0])
ax.set_title("Star")
ax = sns.histplot(df[df["class"] == 0],y="redshift", ax=axis[1])
ax.set_title("Galaxy")
ax = sns.histplot(df[df["class"] == 1],y="redshift", ax=axis[2])
ax.set_title("QSO")
#plt.show()
print(df["class"].value_counts())
X = df.drop("class", axis=1)
y = df["class"]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100)
xgb.fit(X_train, y_train)
p_red = xgb.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("confusion matrix: \n", confusion_matrix(y_test, p_red))
print("accuracy score: \n", accuracy_score(y_test, p_red))
print(classification_report(y_test, p_red))
params = {
    "n_estimators": [100,200,300,500],
    "learning_rate": [0.01,0.1],
    "max_depth": [5,8,12,20,30],
    "colsample_bytree": [0.3,0.4,0.5,0.8,1]
}
from  sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=XGBClassifier(), param_grid=params, cv=5,n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)
y_pred = grid.predict(X_test)
print("confusion matrix: \n", confusion_matrix(y_test, p_red))
print("accuracy score: \n", accuracy_score(y_test, p_red))
print(classification_report(y_test, p_red))
