import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from narwhals.selectors import categorical

from DecisionTreeClassifier import col_names

df = pd.read_csv(r"C:\Users\ozmir\Downloads\14-income_evaluation.csv")
print(df.head())
print(df.shape)
print(df.columns)
col_names = ["age","workclass","finalweight","education",
             "education_num","marital_status","occupation",
             "relationship","race","sex","capital_gain",
             "capital_loss","hours_per_week","native_country",
             "income"]
df.columns=col_names
print(df.columns)
print(df.describe())
print(df.isnull().sum())

categorical = [col for col in df.columns if df[col].dtype == "O"]
numerical = [col for col in df.columns if df[col].dtype != "O"]
print(categorical)
print(df[categorical].head())
for col in categorical:
    print(df[col].value_counts())
print(df["income"].value_counts())

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.countplot(x="income",hue="sex",data=df)
ax.set_title("Distribution of income with gender")
plt.show()