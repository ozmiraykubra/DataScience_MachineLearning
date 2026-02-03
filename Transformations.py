import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['CRIM','ZN','INDUS','CHAS','NDX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT','MEDV']
df = pd.read_csv(r"C:\Users\ozmir\Downloads\23-boston.csv", header=None, delimiter=r"\s+",names=column_names)
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.describe())
import math

def plot_all_histograms(df, title_prefix = ""):
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    print(num_cols)

    plt.figure(figsize=(n_rows * 4, n_cols * 5))

    for i, col in enumerate(num_cols,1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col],kde=True,bins=10)
        plt.title(f"{title_prefix} {col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    #plt.show()

plot_all_histograms(df, title_prefix="Original - ")


from scipy.stats import skew
X = df.drop("MEDV", axis=1)
y = df["MEDV"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
from sklearn.preprocessing import PowerTransformer
pt_X = PowerTransformer(method = "yeo-johnson")
X_train_transformed = pt_X.fit_transform(X_train)
X_test_transformed = pt_X.transform(X_test)
column_names = ['CRIM','ZN','INDUS','CHAS','NDX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT']
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=column_names)
print(X_train_transformed_df)
plot_all_histograms(X_train_transformed_df, title_prefix="Transformed - ")
plot_all_histograms(df, title_prefix="Original - ")
from scipy.stats import boxcox
y_train_transformed, lambda_y = boxcox(y_train)
def inverse_boxcox(y, lambda_y):
    if lambda_y == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_y + 1, 1 / lambda_y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_transformed, y_train_transformed)
y_pred_transformed = model.predict(X_test_transformed)
y_pred_original = inverse_boxcox(y_pred_transformed, lambda_y)
print(y_pred_original)
print(y_pred_transformed)

from sklearn.metrics import r2_score, mean_squared_error
print("R2 Score: ", r2_score(y_test, y_pred_original))
print("MEAN SQUARED ERROR: ", mean_squared_error(y_test, y_pred_original))

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 Score: ", r2_score(y_test, y_pred))
print("MEAN SQUARED ERROR: ", mean_squared_error(y_test, y_pred))

