import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
print(df)
#df.dropna(inplace=True)

#imputation
#mean imputation

sns.histplot(data=df["age"], kde=True)
#plt.show()

plt.close('all')

df["Age_mean"] = df["age"].fillna(df["age"].mean())
print(df[["Age_mean", "age"]])

sns.boxplot(data= df , y="age")
#plt.show()

# median
df["Age_median"] = df["age"].fillna(df["age"].median())
print(df[["Age_median","Age_mean","age"]])

#mode imputation -> categorical values
print(df.isnull().sum())
print(df[df["embarked"].isnull()])
print(df["embarked"].unique())