import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.graph_objs.indicator.gauge import threshold
from sqlalchemy.dialects.mssql.information_schema import columns
from xgboost import train

df = pd.read_csv(r"C:\Users\ozmir\Downloads\21-housing.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['ocean_proximity'].value_counts())
sns.histplot(data=df, x = "housing_median_age", kde=True)
#plt.show()
print(df.columns)
columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,12))
fig.suptitle("Distributions", fontsize = 18, fontweight = "bold")
for i,col in enumerate(columns):
    row = i//3
    col_idx = i % 3
    current_ax = axes[row, col_idx]
    sns.histplot(data=df, x=col,kde=True, ax=current_ax, bins=30)
    current_ax.set_title(col,fontsize=10, fontstyle="italic")
plt.tight_layout()
#plt.show()

df.select_dtypes("float64","int64").columns
def find_outliers_iqr(df, threshold=1.5):
    outlier_summary = {}

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    print(numeric_cols)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        print(IQR)

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(len(outliers))
        outlier_summary[col] = {
            "outlier_count": outliers.shape[0],
            "outlier_percentage": 100 * outliers.shape[0] / df.shape[0],
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
        return pd.DataFrame(outlier_summary)
print(find_outliers_iqr(df, threshold=2))

def remove_outliers_from_column(df, target_col, threshold=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def remove_outliers_from_all_columns(df, threshold=1.5):
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        df_clean = df_clean[ (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean.copy()

print("original data shape: ", df.shape)
df_target_clean = remove_outliers_from_column(df, "median_house_value").copy()
print("only target column cleaning shape: ", df_target_clean.shape)
df_all_clean = remove_outliers_from_all_columns(df)
print("all target column cleaning shape: ", df_all_clean.shape)
print(df_target_clean.isnull().sum())
print(df_target_clean.describe())
df_target_clean["total_bedrooms"] = df_target_clean["total_bedrooms"].fillna(df_target_clean["total_bedrooms"].median())
print(df_target_clean.describe())
print(df_target_clean.isnull().sum())
print(df_target_clean["ocean_proximity"].value_counts())
df_target_clean = pd.get_dummies(df_target_clean, columns=["ocean_proximity"],drop_first=True)
print(df_target_clean.head())
X = df_target_clean.drop("median_house_value", axis=1)
y = df_target_clean["median_house_value"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
print(X_train.columns)
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error
def evaluate_model(true, predicted):
    r2 = r2_score(true, predicted)
    mse = mean_squared_error(true, predicted)
    mae = mean_absolute_error(true, predicted)
    return mae, mse , r2_score
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Gradient Boost Regressor": GradientBoostingRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "K Neighbors Regressor": KNeighborsRegressor(),
    "XGBoost Regressor": XGBRegressor()
}
for i in range(len(models)):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])
    print("model performance for Training Set")
    print("Root Mean Squared Error (RMSE): ", model_train_rmse)
    print("Mean Absolute Error (MAE): ", model_train_mae)
    print("R2 Score: ", model_train_r2)

    print("------------------------------")

    print("Root Mean Squared Error (RMSE): ", model_test_rmse)
    print("Mean Absolute Error (MAE): ", model_test_mae)
    print("R2 Score: ", model_test_r2)

    print("--------------------------------")
    print("\n")

xgboost_params = {
    "learning_rate": [0.1,0.01],
    "max_depth": [5,8,12,20,30],
    "n_estimators": [100,200,300,500],
    "colsample_bytree": [0.3,0.4,0.5,0.7,1],
}
from sklearn.model_selection import RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=XGBRegressor(),param_distributions=xgboost_params,cv=5,n_jobs=-1)
randomized_cv.fit(X_train, y_train)
print(randomized_cv.best_params_)
model = XGBRegressor(n_estimators=300,max_depth=20,colsample_bytree=0.7)
model.fit(X_train, y_train)
