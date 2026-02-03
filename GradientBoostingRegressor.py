import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\ozmir\Downloads\18-concrete_data.csv")
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.corr())
sns.scatterplot(data=df, x="Cement", y="Strength", hue="Age")
#plt.show()

# dependent feature -> Strength
X = df.drop("Strength",axis = 1)
y = df["Strength"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state = 15)
from sklearn.tree import DecisionTreeRegressor
# first weak learner
tree_reg1 = DecisionTreeRegressor(max_depth=3)
tree_reg1.fit(X_train, y_train)
y2 = y_train - tree_reg1.predict(X_train)
print(y2[:5])
# second weak learner
tree_reg2 = DecisionTreeRegressor(max_depth=40)
tree_reg2.fit(X_train,y2)
y3 = y2-tree_reg2.predict(X_train)
print(y3[:5])
# third weak learner
tree_reg3 = DecisionTreeRegressor(max_depth=40)
tree_reg3.fit(X_train,y3)
y4 = y3-tree_reg3.predict(X_train)
print(y4[:5])
y_pred_total = 0
y_pred = sum(tree.predict(X_test) for tree in (tree_reg1,tree_reg2,tree_reg3))
print(y_pred)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))