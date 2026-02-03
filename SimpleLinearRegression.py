import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:\Users\ozmir\Downloads\1-studyhours.csv")
print(df.head())
print(df.info())
print(df.describe())
plt.scatter(df["Study Hours"], df["Exam Score"])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
#plt.show()

plt.close()

# indepenedent and dependent features
X = df[["Study Hours"]]
Y = df["Exam Score"]
print(type(X))
print(type(Y))

# test-train split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

# standardization the data set
from sklearn.preprocessing import StandardScaler
print(df.head())

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train )

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)
print("Coefficient: ", regression.coef_ )
print("Intercept: ", regression.intercept_)

plt.scatter(X_train , Y_train)
plt.plot(X_train, regression.predict(X_train),"r")
#plt.show()

plt.close()

# x = 20 , y = ?
print(regression.predict([[20]]))
print(scaler.transform([[20]]))
print(regression.predict(scaler.transform([[20]])))

# prediction with test data
Y_pred_test = regression.predict(X_test)
plt.scatter(Y_pred_test,Y_test)
#plt.show()

plt.close()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_pred_test)
mae = mean_absolute_error(Y_test, Y_pred_test)
rmse = np.sqrt(mse)
print("mse: ", mse)
print("mae: ", mae)
print("rmse: ", rmse)

r2 = r2_score(Y_test, Y_pred_test)
print("r2 score: ", r2)

# adjusted r2 score
print(1-(1-r2)*(len(Y_test)/len(Y_test)-X_test.shape[1]-1))

#-------------MultipleLinearRegression--------------
df = pd.read_csv(r"C:\Users\ozmir\Downloads\2-multiplegradesdataset.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
sns.pairplot(df)
#plt.show()
print(df.corr())
plt.scatter(df['Study Hours'], df['Exam Score'], color = "r")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
#plt.show()

plt.close()
print(df.corr())
sns.regplot(x = df['Study Hours'], y = df['Exam Score'])
#plt.show()

plt.close()
print(df.tail())

# independent and dependent features
X = df[["Study Hours","Sleep Hours","Attendance Rate","Social Media Hours"]]
Y = df["Exam Score"]

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=15)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)
X_test[0]
df.iloc[0]
new_student = [[5,7,90,4]]
new_student_scaled = scaler.transform(new_student)
print(regression.predict(new_student_scaled))

# prediction
y_pred = regression.predict(X_test)
print(y_pred)
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error,r2_score
mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
print("mse: ", mse)
print("mae: ", mae)
score = r2_score(Y_test, y_pred)
print("r2 score: ", score)
residuals = Y_test - y_pred
print(residuals)
sns.displot(residuals, kind="kde")
#plt.show()
print(regression.intercept_)
print(regression.coef_)
print(X.head())
