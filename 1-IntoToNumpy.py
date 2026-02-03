import numpy as np
from numpy.ma.core import masked_print_option

my_list = [1, 2, 3]
print(type(my_list))

np.array(my_list)
my_numpy_array = np.array(my_list)
print(type(my_numpy_array))
print(my_numpy_array.max())
print(my_numpy_array.min())
print(my_numpy_array.mean())
print(my_numpy_array.std())

print(np.ones(5))
print(np.zeros(5))
print(np.random.random(5))

#array arithmetic
my_list1 = [1,2]
my_list2 = [2,3]
print(my_list1+my_list2)

my_numpy_array1 = np.array(my_list1)
my_numpy_array2 = np.array(my_list2)
print(my_numpy_array1+my_numpy_array2)
print(my_numpy_array1*my_numpy_array2)

other_array = np.array([10,20,30,40,50])
print(other_array.min())
print(other_array.max())
print(other_array.sum())
print(other_array.mean())
print(other_array.std())

# arrange & indexing
print(list(range(0,10)))
print(np.arange(0,10))

np_array = np.arange(0,10)
print(np_array)
print(np_array[0])
print(np_array[-1])
print(np_array[1:4:])
print(np_array[::-1])
print(np_array[::2])
print(np_array[2:6:2])

#random
np.random.randn(4)
print(np.random.randn(4,4))
print(np.random.randint(1,300,5))

my_matrix = [[5,10],[15,20]]
print(my_matrix)
print(my_matrix[0][0])
print(my_matrix[0])

# row * column
numpy_matrix = np.array([[5,10],[15,20]])
print(numpy_matrix)
print(numpy_matrix[1][0])
print(numpy_matrix.sum())
print(np.ones(5))
print(np.ones((4,2)))
print(np.ones((3,5)))
print(np.random.random((3,2)))

#matrix arithmetic
first_array = np.array([[10,20],[30,40]])
second_array = np.array([[5,15],[25,35]])
print(first_array+second_array)
third_array = np.array([[50,60]])
print(first_array + third_array)
fourth_array = np.array([[10,20,30,40,50]])
#print(first_array + fourth_array)
print(first_array * 2)
print(first_array / 4)
third_array = np.array([[10],[20]])
print(third_array)
print(third_array.shape)
print(first_array.shape)
print(first_array + third_array)

# matrix multiplication
print(first_array)
print(second_array)
print(first_array*second_array)

first_matrix = np.array([[10,20,30]])
print(first_matrix.shape)
second_matrix = np.array([[2,3],[2,3],[2,3]])
print(second_matrix.shape)

result_matrix = first_matrix.dot(second_matrix)
print(result_matrix)
print(result_matrix.shape)

new_array = np.random.randint(1,100,20)
print(new_array)
print(new_array > 25)
print(new_array[new_array > 25])
print(new_array[new_array < 25])

# transpose & reshape
matrix_array = np.array([[10,20],[20,30],[30,40]])
print(matrix_array)
print(matrix_array.shape )
print(matrix_array.transpose())
print(matrix_array.T)
random_array = np.random.random((6,1))
print(random_array)
print(random_array.shape)
print(random_array.reshape(2,3))
print(random_array.reshape(3,2))

#z-score
data = np.array([10,12,13,15,18,25,100,105])
# ooutlier
mean = np.mean(data)
print(mean)
print(data.sum())
std = np.std(data)
print(std)
z_score = (data - mean) / std
print(z_score)
print(z_score > 1)
print(z_score[z_score > 1])
print(data[z_score > 1])

#math equations
# 2x + 3y = 8 and 5x + 9y = 10
#coefficient
A = np.array([[2,3],[5,9]])
#constant
b = np.array([8,10])
solution = np.linalg.solve(A,b)
print(solution)