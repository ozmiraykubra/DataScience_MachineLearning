import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# email classification
df = pd.read_csv(r"C:\Users\ozmir\Downloads\9-email_classification_svm.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())
sns.scatterplot(x=df["subject_formality_score"], y=df["sender_relationship_score"], hue=df["email_type"])
#plt.show()
print(df["email_type"].value_counts())
X = df.drop("email_type", axis=1)
y = df["email_type"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
print(X_train)
from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)
print(svc.coef_)
y_pred = svc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
rbf = SVC(kernel="rbf")
rbf.fit(X_train, y_train)
y_pred = rbf.predict(X_test)
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))

df = pd.read_csv(r"C:\Users\ozmir\Downloads\9-loan_risk_svm.csv")
print(df.isnull().sum())
print(df.info())
sns.scatterplot(x=df["credit_score_fluctuation"], y=df["recent_transaction_volume"],hue=df["loan_risk"])
#plt.show()
X = df.drop("loan_risk", axis=1)
y = df["loan_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
linear =SVC(kernel="linear")
linear.fit(X_train, y_train)
y_pred3 = linear.predict(X_test)
print(classification_report(y_pred3, y_test))
print(confusion_matrix(y_pred3, y_test))

rbf =SVC(kernel="rbf")
rbf.fit(X_train, y_train)
y_pred4 = rbf.predict(X_test)
print(classification_report(y_pred4, y_test))
print(confusion_matrix(y_pred4, y_test))

df = pd.read_csv(r"C:\Users\ozmir\Downloads\9-seismic_activity_svm.csv")
print(df.head())
print(df.describe())
print(df["seismic_event_detected"].value_counts())
sns.scatterplot(x=df["underground_wave_energy"], y=df["vibration_axis_variation"], hue=df["seismic_event_detected"])
#plt.show()
# manuel rbf kernel
print(df.columns)
df["underground_wave_energy_sq"] = df["underground_wave_energy"]**2
df["vibration_axis_variation_sq"] = df["vibration_axis_variation"]**2
df["interaction"] = (df["underground_wave_energy"] * df["vibration_axis_variation"])
print(df.head())
X = df.drop("seismic_event_detected", axis=1)
y = df["seismic_event_detected"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
import plotly.express as px
fig = px.scatter_3d(df,
    x="underground_wave_energy_sq",
    y="vibration_axis_variation_sq",
    z="interaction",
    color="seismic_event_detected")
#fig.show()