import numpy as np
import pandas as pd

weather_df = pd.read_excel('C:/Users/ozmir/Downloads/6-weather.xlsx')
print(weather_df)
print(weather_df.head())
print(weather_df.tail())
print(weather_df.info())
print(weather_df.describe())
print(weather_df.count())
print(weather_df.isna())

# working with missing data
weather_na_df = pd. read_excel('C:/Users/ozmir/Downloads/6-weatherna.xlsx',)
print(weather_na_df)
print(weather_na_df.describe())
print(weather_na_df['Paris'].count())
print(weather_na_df['Paris'].isna())
print(weather_na_df.dropna())
print(weather_na_df.dropna(axis=0))
print(weather_na_df.drop('Paris', axis=1))
print(weather_na_df.fillna(22))
print(weather_na_df.mean())
print(weather_na_df.median())
print(weather_na_df.isnull().sum())
print(weather_na_df.fillna(weather_na_df.mean()))

#group by
df = pd.read_csv('C:/Users/ozmir/Downloads/6-employee.csv')
print(df)
print(df.describe())
print(df['Salary'].mean())
print(df[['Salary','Experience']].mean())
print(df.count())
print(df[df['Experience']>6].count())
print(df.head())
df_grouped = df.groupby("Department")
print(df_grouped.count())
print(df_grouped.describe())
print(df_grouped['Salary'].mean())

df_grouped2 = df.groupby("City")
print(df_grouped2['Salary'].mean())

# ------------------DataFramesConcatMerge-------------------
df1 = pd.read_csv('C:/Users/ozmir/Downloads/7-concat_data1.csv')
print(df1)

df2 = pd.read_csv('C:/Users/ozmir/Downloads/7-concat_data2.csv')
print(df2)

#concat
df_concat = pd.concat([df1, df2],ignore_index=True)
print(df_concat)

# merge
df_merge1 = pd.read_csv('C:/Users/ozmir/Downloads/7-merge_data1.csv',)
print(df_merge1)

df_merge2 = pd.read_csv('C:/Users/ozmir/Downloads/7-merge_data2.csv',)
print(df_merge2)

# merge - outer join
df_merged_outer = pd.merge(df_merge1, df_merge2, how='outer', on='Employee_ID')
print(df_merged_outer)

# merge - left join
df_merged_left = pd.merge(df_merge1, df_merge2, how='left', on='Employee_ID')
print(df_merged_left)

# merge - right join
df_merged_right = pd.merge(df_merge1, df_merge2, how='right', on='Employee_ID')
print(df_merged_right)

# merge - inner join
df_merged = pd.merge(df_merge1, df_merge2, how='inner', on='Employee_ID')
print(df_merged)

#-----------DataFrameApplys----------------
df = pd.read_csv('C:/Users/ozmir/Downloads/8-apply_function_data.csv')
print(df)

def salary_category(salary):
    if salary < 50000:
        return "Low"
    elif 50000 <= salary < 80000:
        return "Medium"
    else:
        return "High"
print(df["Salary"].apply(salary_category))
df["salary_Category"] = df["Salary"].apply(salary_category)
print(df)

def new_performance_score(Performance_Score):
    if Performance_Score > 12:
        return Performance_Score+1
    else:
        return Performance_Score
df["New_Performance_Score"] = df["Performance_Score"].apply(new_performance_score)
print(df)

def adjust_performance(experience):
    if experience > 10:
        return 1
    else:
        return 0
print(df["Experience"].apply(adjust_performance))

df["Adjusted"] = df["Experience"].apply(adjust_performance)
df["New_Score"] = df["Performance_Score"] + df["Adjusted"]
print(df)

def adjust_new(row):
    if row["Experience"] > 10:
        return row["Performance_Score"] + 1
    else:
        return row["Performance_Score"]
df["Adjusted_Score"] = df.apply(adjust_new, axis=1)
print(df)

df["Formatted_Name"] = df["Name"].apply(lambda x : x.replace("_"," "))
print(df)