import pprint

import pandas as pd
import numpy as np

data= np.random.randn(4,3)
print(data)
data_frame = pd.DataFrame(data)
print(data_frame)
print(type(data_frame))
print(data_frame[0])

new_df = pd.DataFrame(data, index = ["Atil", "Ali","Ahmet","Atlas"], columns = ["Salary", "Age" , "Seniority"])
print(new_df)
print(new_df["Age"])
print(new_df["Salary"])
print(new_df[["Age","Salary"]])
print(new_df.loc["Atlas"])
print(new_df.iloc[:,1])

new_df["Extra"] = 10
print(new_df)

new_df.drop("Extra",axis=1, inplace=True)
print(new_df)

new_df.loc["Atlas","Salary"] = 2000
print(new_df)

print(new_df>0)
print(new_df[new_df>0])
print(new_df[new_df["Salary"]>0])
reset_frame = new_df.reset_index()
print(reset_frame)
print(reset_frame.loc[0])

new_indices = ["Atl", "Al","Ahmt","Atls"]
new_df["NewIndex"] = new_indices
print(new_df)
new_df.set_index("NewIndex", inplace=True)
print(new_df)
print(new_df.loc["Atl"])

#multi index
first_index = ["Simpson","Simpson","Simpson","South Park","South Park","South Park"]
inner_index = ["Homer", "Bart", "Marge", "Cartman","Kenny","Kyle"]
zipped_index = list(zip(first_index, inner_index))
pprint.pprint(zipped_index)
zipped_index = pd.MultiIndex.from_tuples(zipped_index)
print(zipped_index)

sample_values = np.ones((6,2))
big_df = pd.DataFrame(sample_values, index=zipped_index,columns=["Age","Salary"])
print(big_df)
print(big_df["Age"])
print(big_df.loc["Simpson"])
print(big_df.loc["Simpson"].loc["Homer"])

