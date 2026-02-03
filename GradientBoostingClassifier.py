import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.graph_objs.indicator.gauge import threshold
from sqlalchemy.dialects.mssql.information_schema import columns
df = pd.read_csv(r"C:\Users\ozmir\Downloads\19-heart.csv")
print(df.head())
print(df.info())
print(df['target'].value_counts())
df.hist(bins=40, figsize=(15,10))
#plt.show()
X = df.drop('target', axis=1)
y=df['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
def correlation_for_dropping(df, threshold):
    columns_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
               colname = corr.columns[i]
               columns_to_drop.add(colname)
    return columns_to_drop
correlation_for_dropping(X_train, 0.80)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#hyperparameter tuning
parameters = {
    "loss": ['log_loss','exponential'],
    "learning_rate": [0.01,0.05,0.1],
    "n_estimators": [100,150,180,200],
    "max_depth": [3,4,5],
    "subsample": [0.8,1.0]
}
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parameters, cv=5, n_jobs=-1, verbose=1)
print(grid_search.fit(X_train, y_train))
