import matplotlib
import numpy as np
import matplotlib.pyplot as plt

age_list = [10,20,30,30,30,40,50,60,70,75]
weight_list = [20,60,80,85,86,87,70,90,95,90]

plt.plot(age_list, weight_list,"r")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.title("Age vs Weight")
#plt.show()

#numpy
np_age_list = np.array(age_list)
print(np_age_list)
np_weight_list = np.array(weight_list)
plt.plot(np_age_list, np_weight_list,"g")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.title("Age vs Weight")
#plt.show()

numpy_arr1 = np.linspace(0,10,20)
print(numpy_arr1)

numpy_arr2 = numpy_arr1 ** 3
print(numpy_arr2)

plt.plot(numpy_arr1, numpy_arr2,"b--")
#plt.show()

plt.subplot(1,2,1)
plt.plot(numpy_arr1, numpy_arr2, "r*-")
plt.subplot(1,2,2)
plt.plot(numpy_arr2, numpy_arr1, "g--")
#plt.show()

my_figure = plt.figure()
figure_axes = my_figure.add_axes([0.1,0.1,0.3,0.3])
figure_axes.plot(numpy_arr1, numpy_arr2,"g")
figure_axes.set_xlabel("X Axis")
figure_axes.set_ylabel("Y Axis")
figure_axes.set_title("Graph Title")
#plt.show()

plt.close('all')

new_fig = plt.figure(dpi=100)
new_axes = new_fig.add_axes([0.1,0.1,0.9,0.9])
new_axes.plot(numpy_arr1, numpy_arr1 ** 2, label = "numpy array ** 2")
new_axes.plot(numpy_arr1, numpy_arr1 ** 3, label = "numpy array ** 3")
new_axes.legend()
#plt.show()

plt.close('all')

#---------------MatplotlibStyles--------------
data1 = np.linspace(0,10,20)
print(data1)
data2 = data1 ** 2
print(plt.subplots())
my_fig, my_axes = plt.subplots()
print(type(my_fig))
print(type(my_axes))
my_fig, my_axes = plt.subplots()
my_axes.plot(data1, data2 )
my_axes.plot(data2, data1)
#plt.show()

plt.close('all')

(new_fig, new_axes) = plt.subplots()
new_axes.plot(data1, data1 + 2, color ="blue", linewidth = 2 )
new_axes.plot(data1, data1 + 4, color ="yellow", linewidth = 0.5 )
new_axes.plot(data1, data1 + 8, color ="red", linewidth = 1, linestyle="-." )
new_axes.plot(data1, data1 + 16, color ="green",linestyle = ":", marker="p", markersize=4, markerfacecolor="red" )
#plt.show()

plt.close('all')

# scattero
# plt.scatter(data1, data2, color="orange")
#plt.show()

plt.close('all')

# histogram
new_arr = np.random.randint(0,100,50)
print(new_arr)

#plt.hist(new_arr)
#plt.show()

# box plot
plt.boxplot(new_arr)
plt.show()