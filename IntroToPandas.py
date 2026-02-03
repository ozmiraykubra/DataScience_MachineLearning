import numpy as np
import pandas as pd

grades = {"Atil":50, "James":60, "Lars": 30}
print(pd.Series(grades))

names = ["Atil", "James", "Lars"]
grades = [50,60,30]
print(pd.Series(names))
print(pd.Series(grades))
print(pd.Series(grades,names))
print(pd.Series(names,grades))
print(pd.Series(data=grades , index=names))

#with numpy
numpy_array = np.array([50,40,30,20])
print(pd.Series(numpy_array))

#arithmetic
contest_result = pd.Series(data=[10,5,100], index=["Atil","James","Lars"])
contest_result2 = pd.Series(data=[20,50,10], index=["Atil","James","Lars"])
print(contest_result)
print(contest_result["Atil"])
print(contest_result2["James"])
final_result = contest_result + contest_result2
print(final_result)
print(contest_result*contest_result2)
print(contest_result/contest_result2)
print(contest_result-contest_result2)

different_series = pd.Series([20,30,40,50],["a","b","c","d"])
print(different_series)
different_series2 = pd.Series([10,5,3,1],["a","c","f","g"])
print(different_series2)
print(different_series+different_series2)
