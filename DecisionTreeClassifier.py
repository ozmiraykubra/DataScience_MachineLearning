import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unicodedata import category

from NaiveBayesClassifier import label_encoder

df = pd.read_csv(r"C:\Users\ozmir\Downloads\13-car_evaluation (1).csv")
print(df.head())
print(df.shape)
col_names = ["buying","maint","doors","persons","lug_boot","safety","class"]
df.columns = col_names
print(df.head())
print(df.info())
for col in df.columns:
    print(df[col].value_counts())
print(df["doors"].unique())
df["doors"] = df["doors"].replace('5more','5')
print(df["doors"].unique())
df['doors'] = df['doors'].astype(int)
print(df["persons"].unique())
df["persons"] = df["persons"].replace('more','5')
print(df["persons"].unique())
df['persons'] = df['persons'].astype(int)
print(df.info())
sns.scatterplot(x=df["buying"], y=df["maint"], hue=df["class"])
#plt.show()
sns.barplot(x=df["buying"], y=df["maint"], hue=df["class"])
#plt.show()
X = df.drop(["class"], axis=1)
y = df["class"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_cols = ["buying","maint","lug_boot","safety"]
numerical_cols = ["doors","persons"]

ordinal_encoder = OrdinalEncoder(categories=[
    ["low","med","high","vhigh"],
    ["low","med","high","vhigh"],
    ["small","med","big"],
    ["low","med","high"]
])

preprocessor = ColumnTransformer(transformers=[
    ('transformation_name_doesnt_matter',ordinal_encoder,categorical_cols),
],remainder='passthrough')

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
print(pd.DataFrame(X_train_transformed))

for col in numerical_cols:
    print(df[col].value_counts())

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)
tree_model.fit(X_train_transformed, y_train)
y_pred = tree_model.predict(X_test_transformed)
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
plt.figure(figsize = (12,8))
from sklearn import tree
col_names = categorical_cols + numerical_cols
tree.plot_tree(tree_model.fit(X_train_transformed, y_train),feature_names=col_names)
#plt.show()
plt.close()
#hypermarameter tuning
param =  {
    "criterion":["gini","entropy","log_loss"],
    "splitter":["best","random"],
    "max_depth":[1,2,3,4,5,15,None],
    "max_features":["sqrt","log2",None]
}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator = DecisionTreeClassifier(),param_grid=param, cv=5, scoring="accuracy")
import warnings
warnings.filterwarnings("ignore")
grid.fit(X_train_transformed, y_train)
print(grid.best_params_)
y_pred = grid.predict(X_test_transformed)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
tree_model_new = DecisionTreeClassifier(criterion="entropy", max_depth=None, max_features=None, splitter="best")
tree_model_new.fit(X_train_transformed, y_train)
y_pred = tree_model_new.predict(X_test_transformed)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
tree.plot_tree(tree_model_new.fit(X_train_transformed, y_train),feature_names=col_names)
#plt.show()
df_new = pd.read_csv(r"C:\Users\ozmir\Downloads\11-iris.csv")
X = df_new.drop(["Id","Species"], axis=1)
y = df_new["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=10)
tree_model=DecisionTreeClassifier()
tree_model.fit(X_train,y_train)
tree.plot_tree(tree_model.fit(X_train,y_train),feature_names=X_train.columns,filled=True)
#plt.show()
plt.close()