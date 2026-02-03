import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\ozmir\Downloads\athlete_events.csv.zip")
print(data.describe())
print(data.head())
print(data.isnull())
print(data.info())

plt.scatter("Height", "Weight",data=data)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Athlete Height vs Weight")
#plt.show()

sns.set_style("whitegrid")
sns.scatterplot(x="Height", y="Weight",hue="Sex", data=data)
plt.xlabel("Height of Athletes")
plt.ylabel("Weight of Athletes")
plt.title("Athletes Height vs Weight")
#plt.show()

print(data["Medal"].unique())
print(data["Sex"].unique())

plt.close()

sns.set_style("whitegrid")
sns.lineplot(x="Height", y="Weight",hue="Sex", data=data)
plt.xlabel("Height of Athletes")
plt.ylabel("Weight of Athletes")
plt.title("Athletes Height vs Weight")
#plt.show()

plt.close()

sns.set_style("dark")
sns.displot(x="Height", hue="Sex", data=data)
plt.ylabel("Frequency")
plt.title("Athletes Height Distribution")
#plt.show()

plt.close()

sns.set_style("dark")
sns.displot(x="Height", hue="Sex", data=data, kind="kde")
plt.ylabel("Frequency")
plt.title("Athletes Height Distribution")
#plt.show()

sns.barplot(x="Medal", y="Height", hue="Sex" ,data=data)
plt.title("Medals by Height")
#plt.show()

sns.heatmap(data.corr(numeric_only=True))
#plt.show()

plt.close()

#----------BoxPlotExercise------------
# box plot
# median, quartile, min,max,outlier

data = np.array([5,7,9,15,20,22,30,32,35,37,40,50,55,60,100])
plt.figure(figsize=(6,5))
sns.boxplot(y=data)
plt.title("Box Plot")
plt.ylabel("Data Value")
plt.grid(True)
#plt.show()

plt.close()

df = sns.load_dataset("titanic")
print(df)

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(x="class", y="age", data=df)
plt.title("Age by Class")
plt.xlabel("Class")
plt.ylabel("Age")

plt.subplot(1,2,2)
sns.boxplot(x="class", y="fare", data=df)
plt.title("Fare by Class")
plt.xlabel("Class")
plt.ylabel("Fare")

plt.tight_layout()
plt.show()