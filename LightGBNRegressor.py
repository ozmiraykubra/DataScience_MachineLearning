import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from RandomForestClassifier import categorical
from Transformations import y_train_transformed, y_pred_transformed

df = pd.read_csv(r"C:\Users\ozmir\Downloads\24-medical_cost.csv")
print(df.head())
print(df.info())
print(df.describe())
sns.countplot(data=df, x="sex")
#plt.show()
sns.countplot(data=df, x="smoker")
#plt.show()
sns.countplot(data=df, x="region")
#plt.show()
print(df.head())
sns.scatterplot(data=df, x="age", y="charges", hue="sex" )
#plt.show()
sns.histplot(data=df, x="bmi", kde=True)
#plt.show()
sns.histplot(data=df, x="charges", kde=True)
#plt.show()

print(df.head())
df.drop("Id",inplace=True, axis=1)
print(df.head())
df["sex"] = df["sex"].map({"male":0, "female":1})
df["smoker"] = df["smoker"].map({"no":0, "yes":1})
print(df["sex"].value_counts())
print(df["smoker"].value_counts())
print(df.info())

plt.close()

# one hot encoding -----> region
X = df.drop("charges", axis=1)
y = df["charges"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
categorical_cols = ["region"]
preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)],remainder="passthrough")
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print(X_train)
from lightgbm import LGBMRegressor
model = LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# hyperparameter tuning
param_grid = {
    "num_leaves" : [31, 50 , 70],
    "max_depth" : [-1 , 5 , 10],
    "learning_rate" : [0.01, 0.05, 0.1],
    "n_estimators" : [100, 300, 1000],
    "min_child_samples" : [30 , 50 , 70],
    "subsample" : [0.6 , 0.8 , 1.0],
    "colsample_bytree" : [0.6 , 0.8 , 1.0],
    "reg_alpha" : [0 , 0.5 , 1.0],
    "reg_lambda" : [0 , 0.5 , 1.0]
}
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")
random_search = RandomizedSearchCV(
    estimator=LGBMRegressor(verbosity=-1),
    param_distributions=param_grid,
    cv=5,
    verbose=0,
    random_state=15,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
random_search.fit(X_train, y_train)
print(random_search.best_params_)
y_pred = random_search.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# transformation on the dependent feature
from scipy.stats import boxcox
y_train_transformed, lambda_y = boxcox(y_train)
model = LGBMRegressor()
model.fit(X_train, y_train_transformed)
y_pred_transformed = model.predict(X_test)
print(r2_score(y_pred_transformed, y_pred))
print(mean_squared_error(y_pred_transformed, y_pred))
y_pred_transformed = model.predict(X_test)
# inverse Box-Cox
def inverse_boxcox(y, lambda_):
    if lambda_ == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_ + 1, 1 / lambda_)
y_pred_original = inverse_boxcox(y_pred_transformed, lambda_y)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
