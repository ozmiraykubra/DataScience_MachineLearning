import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.f2py.crackfortran import param_eval

df = pd.read_csv(r"C:\Users\ozmir\Downloads\8-fraud_detection.csv")
print(df.columns)
print(df.head())
print(df['is_fraud'].value_counts())

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]
sns.scatterplot(x = X["transaction_amount"], y = X["transaction_risk_score"], hue=y)
#plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
penalty = ["l1", "l2","elasticnet"]
c_values = [100,10,1.0,0.1,0.01]
solver = ["newton-cg","lbfgs","liblinear","sag","saga", "newton-cholesky"]
class_weight = [{0:w , 1:y} for w in [1,10,50,100] for y in [1,10,50,100]]
print(class_weight)
params = dict(penalty=penalty, C=c_values , solver=solver, class_weight=class_weight)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
cv = StratifiedKFold()
grid = GridSearchCV(model, param_grid=params, scoring ="accuracy")
import warnings
warnings.filterwarnings("ignore")
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("score: ", accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
print("confusion_matrix: \n", confusion_matrix(y_pred, y_test))

# roc, auc
model_prob = grid.predict_proba(X_test)  #probabilities for the positive (fraud) class
print(model_prob)
from sklearn.metrics import roc_curve,roc_auc_score
model_auc = roc_auc_score(y_test, model_prob[:,1])
print(model_auc)
# model false positive rate
#model true positive rate
model_auc = roc_curve(y_test_model , model_prob)
print(model_auc)
model_fpr, model_tpr , tresholds = roc_curve(y_test,model_prob)
plt.plot(model_fpr, model_tpr, marker = ".", label="Logistic")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.show()

fig, ax = plt.subplot(figsize = (20 , 10))

ax.plot(model_fpr, model,tpr, marker=".", label="Logistic")

for fpr, tpr ,tresh in zip(model_fpr, model_tpr, tresholds):
    ax.annotate(f"{np.round(thresh, 2)}", (fpr,tpr), ha="center")

ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend()
plt.show()