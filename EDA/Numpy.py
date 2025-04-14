# Numpy ( Numerical Python ) - used for working with arrays, faster that lists. 
'''
import numpy 
a = numpy.array([1,2,3,4,5])
print(a)

# We can also use the alias "np". Alias : are the alternate name for referring the same thing.
import numpy as np
a = np.array([2,4,6,8,10])
print(a)

# checking the verion of the numpy installed 
import numpy
print(numpy.__version__)

# Creating arrays
import numpy as np
idk = np.array(["Hello", "World"])
print(idk)
print(type(idk))'''

# ADDITIIONAL **
'''import numpy as np

arr = np._core.records.fromrecords([(1, "Lion", 20),
                           (2, "Swan", 21)], 
                           names = "Roll number, Name, Age")

print(arr)
print(arr[0])
print(arr[1])
print(arr.Name)
print(arr.Age)'''
# Dimensions in arrays -> is one level of array depth (nested array)
# 0-D Arrays -> aka Scalars, are the elements in an array. Each value in an array is 0-D array.
'''
import numpy as np
a = np.array(7)
print(7)

# 1-D Array -> that has 0-D arrays as its elements
import numpy as np
a = np.array([1,7,5])
print(a)

# 2-D array -> array which has 1-D array as its elements.
import numpy as np
a=np.array([[1,2,4,5],[3,6,7,8]])
print(a)

# 3-D array -> have 2-D arrays as its elements.
import numpy as np
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a)
'''
# Dimensions of array - "(variable_name).ndim"
'''
import numpy as np

hello_0 = np.array(42)
hello = np.array([42]) 
hello_1 = np.array([1,7,5])
hello_2 = np.array([[2,4,6],[1,3,5]])
hello_3 = np.array([[[[1,2], [3,4]], [[5,6],[7,8]]]])

print(hello_0.ndim)
print(hello.ndim)
print(hello_1.ndim)
print(hello_2.ndim)
print(hello_3.ndim)

# Higher dimensional array -> when creating array, we can also define the dimension of array using "ndmin".
import numpy as np

my_array = np.array([7,8,9,2,6], ndmin = 4)
print(my_array)
print("Dimension : ", my_array.ndim)
'''
# Anatomy of an array -
# a) - Axis - it describes the order of the indexing into the array. axis 0 -> 1 D, axis 1 -> 2 D, axis 3 -> 3 D

# b) - Shape - describes the shape of array
'''
import numpy as np
a = np.array([[1,2,3,4,5,6], [2,3,4,7,9,0]])
print(a.shape)

b = np.array([[1,2], [3,4], [5,6]])
print(b.shape)
'''
# To get shape of the array without using numpy
# c) - Rank - gives the rank of array
# d) - datatype 
'''import numpy as np 
a = np.array([1,2,3,4,5,6,7,8,9])
print(a.dtype)'''
# Size of array 
'''
import numpy as np
a = np.array([[1,2,3],[1,2,3]])
print(a.size)'''
# Numpy Array Indexing 
'''
import numpy as np

new = np.array([24,56,78])
print(new[1])
print("The sum of all numbers of array is : ", new[1] + new[0] + new[2])
# using loop
total = 0
for i in range(0, len(new)):
    total = total + new[i]
print(total)
'''
# Access 2D array -
'''
import numpy as np
my_2d = np.array([[1,2], [5,7]])
print(my_2d[1][1])
#Access 3D array -
my_3d = np.array([[[1,2], [3,4]], [[6,7], [8,9]]])
print(my_3d[0,1,1])
print(my_3d[0,-1,-1])
'''
# Array Slicing -> [start : end] and [start : end : step]
'''
import numpy as np
# slicing 1D array
qwerty = np.array([1,2,4,5,6,7,9])

print(qwerty[3:])
print(qwerty[:3])
print(qwerty[1:4])
print(qwerty[3:1])
print(qwerty[1:-4])
print(qwerty[-1:4])
print(qwerty[-3:-1])

print(qwerty[1:4:2])
print(qwerty[2:6:2])

# slicing 2D array

mech_keyboard = np.array([["Shu", "Swan", "Lion"], ["Hello", "World", "Moon"]])
# In 2D array, to access any element, we first need to access the array and then perform the required action.
print(mech_keyboard[0, 0:])
print(mech_keyboard[1, 1:])

# Accessing a particular element of all the arrays -
print(mech_keyboard[0:2, 1])
print(mech_keyboard[0:2, 0:2])
print(mech_keyboard[0:2, 1])

my_love = np.array([["Lion", "Bear", "Moon", "Tuii"], ["Swan", "Panda", "Pari", "Tuii Stucker"]])

print(my_love[0:2, 1])
print(my_love[0:2, 3])
print(my_love[0:2, 1:3])
'''
# Data Types in Numpy
'''
import numpy as np

a = np.array([1,2,3,4,7])
b = np.array(["Hello", "World"])
print("a ka datatype : ", a.dtype)
print("b ka datatype : ", b.dtype)

# creating array with defined datatype -
c = np.array([26,8,9,8], dtype = "S")
print(c)
d = np.array([1,2,3,4,5], dtype = "i8")
print(d.dtype)
e = np.array(["hello", "World"], dtype = "O")
print(e)
print(e.dtype)
'''
# Converting Data Type on Existing Arrays -> the best way is to make copy of the array with "astype()" method. and provide data type as a parameter.
'''
import numpy as np

me = np.array([1.1, 2.7, 3, 4, 5])
new_dtype = me.astype(float) # bool, str, complex, float

print(new_dtype)
print(new_dtype.dtype)

new_list = np.array([[1,2], [3,4]])
bool_list = new_list.astype(bool)
print(bool_list)

my_str_list = np.array([[2,4], [8,6]])
new = my_str_list.astype(str)
print(new)

newbie = np.array([54, 67, 89])
newbie_2 = newbie.astype("complex")
print(newbie_2)
print(newbie_2.dtype)'''

# Ways of creating arrays -
# 1 - np.array()
# 2 - np.fromiter(iterator, dtype)
'''
import numpy as np
baby = [2,5,8,0]
baby2 = np.fromiter(baby, complex)
print(baby2)

my_str = "Hello World"
arr = np.fromiter(my_str, "object")
print(arr.dtype)'''

# 3 - numpy.arange() - gives evenly spaced values in the interval.
'''
import numpy as np
a = np.arange(1, 20, 3, dtype = "i") # dtype can also be defined as - np.int32
print(a) '''
# 4 - np.linspace()
# syntax = np.linspace(start, end, step, dtype, endpoint = True, retstep = False, axis = 0)
'''
import numpy as np
a = np.linspace(1, 10, 3)
b = np.linspace(2, 10, 4, dtype  = np.complex128) # dtype = np.complex is not supported, so use normal complex or complex128
c = np.linspace(1, 10, 5, dtype = "int32", endpoint = True, retstep = False, axis = 1)
print(a)
print(b)
print(c)
'''
# 5 - np.empty() - creates new array by taking the size of the array.
'''
import numpy as np
a = np.empty([3,3], dtype = np.int8)
b = np.empty([3,3], dtype = np.int32, order = 'f')
print("Array A : ", a)
print("Array B : ", b)
'''
# 6 - np.ones( shape, dtype = None, order = "C") - all elements will be 1.
'''
import numpy as np 
a = np.ones([3,3], dtype = np.int32, order = "f")
print(a)
# & 7np.zeroes() - all elements will be 0'''

# Copying the array and View the array
'''import numpy as np

arr = np.array([1,2,3,4,5])

x = arr.copy() # Copy - copies the array 

arr[0] = 20 # updating the array

y = arr.view() # View - view the array after the changement. The view array is updated if any command was given before it. (here arr[o] = 20)

print(x)
print(y)
print(arr)'''

'''import numpy as np

arr = np.array([5,4,2,7,9])

x = arr.copy()
y = arr.view()

# arr.base to check if it own the data or not
# owning the data means that the array is responsible for managing the data. If the array is deleted, the memory is freed.
# In case of not owning the data. here "y.base", the array is just the view of sliced part of any other original array.


print(x.base) # Returns "None" - > Owns the data
print(y.base) # does not owns the data 

# Shape of the Array - 
# It is actually the number of element in each dimension.
# The shape of the array is the number of elements in each dimension. * -> (column, rows)

new = np.array([[1,2,3,4], [5,6,7,8]])
x = new.shape
print(x)

new_2 = np.array([1,2,3,4,5], ndmin = 4)
print(new_2)
print(new_2.shape) # Gives 1,1,1,5 -> 
# dim1 = [[[[1,2,3,4,5]]]] Contains 1 element.(outermost bracket containing all other elements.) , dim2 - [[[1,2,3,4,5]]], dim3 - [[1,2,3,4,5]], dim3 - [1,2,3,4,5]'''

# Reshaping Array -> changing the shape of array.--------

'''import numpy as np

my_arr = np.array([1,2,3,4,5,6,7,8])
# Reshaping 1-D to 2-D array
print(my_arr.reshape(4, 2)) # row x column

new = my_arr.reshape(2,4)
print(new)

# Resahping from 1-D to 3-D

new_arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

shape  = new_arr.reshape(2, 2, 3) # 2x3 ke 2 matrix (row, col)
print(shape)

new = new_arr.reshape(3, 2, 2) # 2x2 ke 3 matrix
print(new)

# We cannot reshape our array in any form. Becasue the product of shape of matrix decides the number of element. Therefore there must be desired number of element available in the array to be transformed.

# Checking if the array returned is copy or view
print(shape.base) # It returns the original array so it is view.

# Unknown Dimension - We can pass -1 in the dimension, numpy will calculate the number for us. Or we can say it is for defining the column which is automatically done by numpy

new = new_arr.reshape(3, 2, -1) # 3 matrix, 2 row, -1(auto col created by numpy)
print(new)
'''

# Flattening the array -> Converting multidimensional array into 1-D array. 

'''import numpy as np

flat_arr = np.array([[1,2,3,4], [5,6,7,8]])

flat = flat_arr.reshape(-1)

print(flat)'''


# Array Iteration
 
#import numpy as np

'''arr = np.array([[[1,2,3], [5,4,3]], [[3,4,5], [3,2,1]]])

for i in arr:
    print(i)
'''
# We cannot Iterate over custom dimensionalized array.
'''my_arr = np.array([1,2,3,4,5], ndmin = 3)

for i in my_arr:
    print(my_arr[i])'''

# To print all the element of the aray -

'''for x in arr: # It iterates over the array (2 element)
    for y in x: # Iteratres over 2 ele of x
        print(y)
        for z in y: # Iterates over each ele of y
            #print(z)
            pass'''

# nditer -> can be used for iterations. 
# During "for" loops, iterating through each array, we need to use "n" 'for' loops, which can be challenging for high dimensionality array.

"""import numpy as np

arr = np.array([1,2,3,4])
arr2 = np.array([[1,2,3,4], [5,6,7,8]])

for x in np.nditer(arr2):
    print(x)

for x in range(len(arr2)):
    print(arr[x])
    '''for y in range(0, x, 3):
        print(arr[y])
'''
# Iterating array with different data type
# "op_dtypes" -> used to pass the datatype to change datatype of elements. It does not change the datatype in-place, it creates new space in the memory. This new space is called "buffer". 
# flags = ['buffered']

my_arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

for i in np.nditer(my_arr, flags = ["buffered"], op_dtypes = ["S"]):
    print(i)

# [:::], :-iterate whole array, :define the range to iterate, :steps to cover the range
for i in np.nditer(my_arr[:, :4:2]):
    print(i)


for i, k in np.ndenumerate(my_arr):
    print(i, k)  # (row, col) value -> is the format of the answer.

new = np.array([1,2,3])
for i, k in np.ndenumerate(new):
    print(i, k)"""

# Array Join 
# Concatenate
# Stack - h, v, d
'''import numpy as np

arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])

new = np.concatenate((arr1, arr2))
print(new)


arr3 = np.array([arr1, arr2])
arr4 = np.array((arr2, arr1))

new = np.concatenate((arr3, arr4), axis = 1)
print(new)'''

# Stack - WE can concatenate two 1-D array along the second axis which would result in putting them one over other.
'''import numpy as np

arr1 = np.array([1,2,3])
arr2 = np.array([5,6,7])

new = np.stack((arr1, arr2), axis = 0)
print(new)

# hstack() - for row, vstack() - for col, dstack() - for depth

h = np.hstack((arr1, arr2))
print(np.array((arr1, arr2)))
v = np.vstack((arr1, arr2))
d = np.dstack((arr1, arr2))

print(f"{h}")
print(f"{v}")
print(f"{d}")
'''

# Splitting Array - returns list

'''import numpy as np 

# 1-D array
arr = np.array([1,2,3,4,5])
new = np.array_split(arr, 2)
print(new)

# The array has been divided into 2 parts 
print(new[0]) # contains all the elements of the first part
print(new[1]) # contains all the elements of the second part

# 2-D array
arr = np.array([[1,2,3,4,5], [9,8,7,6,5]])
new = np.array_split(arr, 3)
print(new) # The third part will contain an empty list of element.
print(arr.shape) # (x, y) -> Represents the dimension of the array. x -> represents the first dimension containing 2 element. y -> represents the second dimentsion containing 5 elements. Also, x * y = number of elements of the array.
print(new[1])
print(new[0])
print(new[2]) # returns empty list

# 3-D array
a = [1,2,3]
b = [4,5,6]
c = [7,8,9]

arr = np.array([a,b,c])
print(arr) # 2-D array

arr = np.array([[a,b], [b,c]])
print(arr) # 3-D array

# splitting 3-D array
new = np.array_split(arr, 3)
print(new[0])
print(new[1])
print(new[2])

arr = np.array([[1,2,3,4], [5,6,7,8]])

# Only has 1 splitter aka hsplit. The new varibale holds arr in the parameter not outside it.
# ** EQUAL DIVISION is supported **
new = np.hsplit(arr, 2)
new = np.hsplit(arr, 4)
#new = np.split(arr, 3) # unqueal division results in Value Error
print(new)'''

# Array Search  - > we can search for a certain value and return the index that get the matches.

# wehre() method is used to search the particular index where the exact number is
# * Returns tuple elements

'''import numpy as np

arr = np.array([7,77,777])
x = np.where(arr == 7)
print(x)

arr = np.array([7,7,7])
x  = np.where(arr == 7)
print(x)

# we can pass our logics in where() as a parameter.
arr = np.array([[2, 4, 6, 8], [10, 11, 14, 15]])
# finding where the values are even.
x = np.where(arr % 2 == 0)

print(x)'''

# Output -> (array([0, 0, 0, 0, 1, 1, 1, 1]), array([0, 1, 2, 3, 0, 1, 2, 3]))   . The first array represents the dimensional position of number of elements where they are even. Therefore, according to the given first array, total 6 elements are even, out of which 4 elements from dimension 1 and next from the dimension 2 ( total 2 elements ). The second array represents all the indexes where even elements are present.

# searchsorted() -> it performs a binary search in the given array and returns the "index", where a particular value should be inserted.
# * usually we use this method on sorted array.
'''import numpy as np

arr1 = np.array([2,4,9,1])
arr = np.sort(arr1)
finding_index = np.searchsorted(arr, 7)

print(finding_index) # total elemetns - 6 (inc 7), so on index 5 is the suitable place to insert 7 (ofc in sorted array).

# we can also perform our search from the right direction

new = np.searchsorted(arr, 7, side = "right")
print(new)

# we can also give an array of element to perform search sorted.

new = np.searchsorted(arr, [2,34,56,78])
print(new)'''

# Numpy Array Filter
# Filtering some elements out of an existing array and creating a new array out of them is called filtering. We filter objects using a boolean index list. If val = True, element in included in the filtered array, else not.

'''import numpy as np

arr = np.array([10, 20, 30, 40, 50])

boolean = [True, False, True, True, False]

new = arr[boolean]

print(new) # Prints only the True values.

# Printing particullar values.
filter_arr = []

for i in arr:
    if i >= 40:
        filter_arr.append(True)
    else:
        filter_arr.append(False)

newarr = arr[filter_arr]

print(newarr)

# Creating filter array directly from the array.

import numpy as np

arr = np.array([77, 877, 778, 80, 99])

filter_arr = arr > 100

newarr = arr[filter_arr]

print(newarr)'''


# * We can directly find the element without actually using loop, using -> Arange()

'''import numpy as np

arr = np.array([2,4,6,8,10,12,14])
my_arr = np.arange(0, 6, 2)
print(my_arr)
'''

# Numpy ufuncs -> "Universal Funcitons", operate on ndarray objects.
# What is vectorization -> Converting iterative statements into vector based operation.

# Create your own ufunc -
# create normal function and then add it into numpy ufunc library.

# frompyfunc() takes the following output -
# function, inputs, outputs

'''import numpy as np
def myadd(x, y):
    return x + y

mydd = np.frompyfunc(myadd, 2, 1)

print(myadd([1,2,3], [5,6,7]))


print(type(np.add))
print(type(np.concatenate))

if type(np.add) == np.ufunc:
    print("It is Universal Function")
else:
    print("It is not Universal Funciton")'''

# Simple Arithmatic 
# add, subtract, multiply, divide, power, absolute, remainder, divmod

"""import numpy as np

arr1 = np.array([10, 20, 30, 40])
arr2 = np.array([90, 80, 70, 60])

# add - add()
new = np.add(arr1, 2) # either 1 parameter or equal to the len of the first array. i.e, either "n" or [x, y, z...,len(arr1)]
print(new)

# subtract - subtact()

new = np.subtract(arr2, arr1)
print(new)

# product - multiply()

new = np.multiply(arr1, arr2)
print(new)

# divide - divide()

new = np.divide(arr2, arr1)
print(new)

# Power - power()

new = np.power([2, 4, 8], 2) # we can also provide list of numbers to assign power to the corresponding elements. - [2, 2, 3] -> ans [4, 16, 212]
print(new)

# Remainder - we can use "mod()" or "remainder()"

new = np.mod(arr1, 3)
print(new)

new = np.remainder(arr1, 3)
print(new)

# Quoteint and Mod -> divmod() gives the quoteint and remainder both of the given array.

new = np.divmod(arr1, 3)
print(new)

# Absolute Values - same as abs()

arr = np.array([[-2, - 5, 10], [2, -3, -7]])
new = np.absolute(arr)
print(new)"""

# Rounding Decimals 

# There are mainly 5 wasy of doing it
# truncation, fix, ceil, floor, rounding

"""import numpy as np

# Truncatiion - trunc() / fix()
# removes the decimal and return the number closest to zero

arr = np.trunc([-3.12455, 7.99])
print(arr) # 7 is more close to zero than 8. therefore it returns 7

# Rounding -> around() -> It increase the preceeding digit of decimal by 1 if the next occuring number is >= 5 else nothing. i.e., 3.166 -> 3.2

arr = np.around([3.1135, 2.1])
print(arr)

# Floor -> rounds the decimal to nearest lower integer.

arr = np.floor([5.55, 7.99, 2.11])
print(arr)

# Ceil -> rounds the integer to nearest upper integer

arr = np.ceil([3.11, 7.99, 2.00])
print(arr)"""



# Numpy Logs -> at base 2, e, 10 or at any custom base

"""import numpy as np

# at base 2 - log2()

arr = np.array([2, 4, 6, 8])
print(np.log2(arr))

# at base 1- - log10()

arr = np.arange(1, 10)
print(np.log10(arr))

# Natural base or base "e" - log()

arr = np.arange(1, 11)
new = np.log(arr)
print("The Natural log of number from 1 to 10 is -")
print(new)

# Log at any base -> numpy does not provide any particular function to find log any base. 

from math import log

new = np.frompyfunc(log, 2, 1,) # log from math always required only two arguments (values and base). Giving more than 2 inputs arguments, the code will show error.

print(new(125, 125))"""

# Numpy Summations -> There is difference between summations and addition. Addition is done between two arguments and summation is done over n elements.

"""import numpy as np

arr1 = [1,2,3,4]
arr2 = [9,8,7,6]

new = np.add(arr1, arr2) # Adds each corresponding elements of the array.
print(new)

new = np.sum([arr1, arr2, arr2, arr1]) # Takes each array as a single element and computes the sum of each array and then add it together too.
# if we provide axis = 1 in the new, then numpy will give the sum of each array as -> [10, 30]
print(new)

# Cumulative sum -> partially adding the elements in the array.
# ex-> [1, 2, 3] = [1, 1+2, 1+2+3] = [1, 3, 7]
new = np.array([2,4,6])
newarr = np.cumsum(new)
print(newarr)

new = np.array([[2,4,6], [3, 5, 7]]) # add all the elements successively.
print(np.cumsum(new))

n = np.cumsum(newarr)
print(n)"""



# Products -> prod()

"""import numpy as np

arr1 = [2,4,6,8]
print(np.prod(arr1)) # 2*4*6*8
print(np.prod([arr1, arr1])) # multiplies the arr1 again or in simple words -> it squares them.

# product over axis 
arr2 = [1,3,5,7]

new = np.prod([arr1, arr2], axis = 1) # multiplies the sigle array elements and other array elements with itself.
print(new)

new2 = np.prod(new)
print(new2) # multiplying 384 and 105

# cumulative product

newarr = np.array([[7,7,7,7],[2,2,2,2]])
print(np.cumprod(newarr)) # 7, 7x7, 7x7x7, 7x7x7x7 and then multiply the resuting with successive 2."""


# Differences -> diff()
# Its called Discrete Difference. It subtracts the successive elements with each other.

"""import numpy as np 

arr = np.array([2, 5, 9, 15])
# 5-2, 9-5, 15, 9
new = np.diff(arr)
print(new)

# * the final array length reduced by 1
# we can also pass how many times the successive array subtracts itself. It can be done by passing n = any integer

new = np.diff(arr, n = 2) # finds the discrete difference and the finds the discrete difference from the resulting array, which inturn reduces the given array 2 times.

print(new)"""


# LCM

"""import numpy as np

a, b = 20, 30

x = np.lcm(a, b)

print(x)

# in case of arry -> we use lcm.reduce()
arr = [20, 30]
print(np.lcm.reduce(arr))"""

# Same for GCD

# Trignometric Functions 
# sin(), cos(), tan() -> takes values in Radians 

"""import numpy as np

x = np.pi

y = np.sin(x/2)
print(y)

# creating array of values
new = np.array([x/2, x/3, x/4])

y = np.sin(new)
z = np.cos(new)
a = np.tan(new)

print(y)
print(z)
print(a)

# Convert Degrees into Randians

arr = np.array([90, 180, 270, 360])

x = np.deg2rad(arr)

print(np.sin(x))  

# Radian to Degrees -> rad2deg(), the new array will contain elements in terms of pi. - 2*np.pi, 3 * np.pi/2, ...


arr = np.array([1,2,3,3,4,5,7,10,24,7])
x = np.where(arr == 7)
print(arr[x[0]])"""


'''import math
import numpy as np
# degree = radian * 180 / pi(II)
arr = np.array([math.pi/2, math.pi/3, math.pi/4])

print("Original array : \n", arr) # pi = 22/7 

# The original array is present in the form of Degree (^o). Thats only why it is convertable into Radian
print(np.rad2deg(arr)) # pi != 22/7, its = II / 2
print(np.deg2rad(arr))

x = np.rad2deg(arr)
print(x)
print(np.ceil(np.tan(x)))

y = np.where(np.tan(arr) <= 1)
print(np.floor(np.tan(arr[y]))) # output == 0, this proves that the value of tan 45^o is not == absolute 1. Its 0.9999...



# Finding Angles -> arcsin(), arccos(), arctan() -> produces radian values of corresponding sin, cos, or tan values.
# jo bhi hum arcsin() or arccos() or arctan() me value provide krenge, uska radian value hume mil jayegi.

import pretty_errors
import numpy as np

arr = np.arcsin(1/2)

print(arr) # gives radian value of pi by 6 == 22/7 * 1/6

y = arr

print(y) # it gives radian value of sin(1)

print(np.rad2deg(arr))

z = np.tan(y)
print()
'''

# finding Hypotenuse

'''import numpy as np

base = 4
perp = 3

hy = np.hypot(base, perp)

print("Hypotenuse :", int(hy))'''
      


# Set Operations 

"""import numpy as np

arr = np.arange(1, 10).reshape(3,3)

arr = np.array([[1,2,3,4], [4,3,6,7]])
# Unique() operation is used to find unique elements of the array.
x = np.unique(arr)

print(x)

# Finding union -> union1d() is used to find the union of the two given array

import numpy as np

arr1 = np.array([1,2,3,4])
arr2 = np.array([2,4,6,8])

x = np.union1d(arr1, arr2)
print(x)

# Finding intersection -> intersect1d() is used to find the intersection of two array. As this is used to find intersection of two array, therefore for interseciton of two 3D or 2D array, we need two 3D or 2D array.

arr1 = np.array([1,2,3,4])
arr2 = np.array([2,4,6,8])

new = np.concatenate(([[arr1, arr2], [arr2, arr1]]), axis = 1)
print(new)

x = np.intersect1d(new, arr1)
print(x)
# we can also pass perameter in intersect1d() - "assume_unique = True". It speeds up the computation, so will be good that always set to be True.

x = np.intersect1d(arr1, arr2, assume_unique = True)
print(x)

# Finding the difference between the arrays -> setdiff1d()

arr1 = np.array([1,2,3,4])
arr2 = np.array([2,4,6,8])

# find the element that are present only in arr1 not in arr2
new = np.setdiff1d(arr1, arr2, assume_unique = True)

# find the element that are persent only in arr2 not in arr1
new2 = np.setdiff1d(arr2, arr1, assume_unique = True)

print(new)
print(new2)

# Symmetric Difference -> used to find the values that are unique in both arrays.
# syntax -> "setxor1d()"

arr1 = np.array([1,2,3,4])
arr2 = np.array([2,4,6,8])

new = np.setxor1d(arr2, arr1, assume_unique = True)
print(new)"""


'''import numpy as np

arr = np.array([1,2,34])
new = np.reshape(3, 1)
print(new)
new = arr

new[2] = 23
print(new)

# appending value in array

new = np.append(arr, [7])
print(new)
'''

# Swapping column in numpy array

'''import numpy as np

my_arr = np.arange(0, 9)
new = my_arr.reshape(3, 3)

print(new)

new[:,[0, 2]] = new[:,[2, 0]]

print(new)

print(new[1:, :2]) # row: , column:
print(new[1:, 1:])

# comparing two array

import numpy as np

my_array = np.array([])'''


# Matrix Manipulation in Python
# It can be implented as 2D Array. 

# Operations in Matrix  -> add(), subtract(), divide(), multiply(), dot(), sqrt(), sum(x, axis), "T" -> argument used to transpose matrix.

'''import numpy as np 

arr1 = np.array([2,4,6,8])
arr2 = np.array([1,3,5,7])'''

'''new = np.add((arr1, arr2), arr2)

print(new)

new = np.subtract(arr1, arr2)
print(new)

arr1 = np.array([[1,2,3,4,5], [6,7,8,9,0]])
print(arr1.reshape(1, 10))

arr2 = np.array([[5,4,3,2,1], [6,34,7,23,8]]).reshape(2, 5)
print(arr2)

print(np.add(arr1, arr2))'''

# Multiply and Dot

'''print(np.multiply(arr1, arr2))
print(arr1.reshape(2,2))

# multiply() -> computes element wise multiplication
print(np.multiply(arr1.reshape(2,2), arr2.reshape(2,2)))''' # The matrix is multiplied with the corresponding element of the array.

# dot() -> computes matrix multiplication rather than element wise.

'''a = arr1.reshape(2, 2)
b = arr2.reshape(2, 2)

x = np.dot(a, b)

print(x)'''

# sqrt() -> computes square root of element of the matrix
# takes one positional argument. If the axis argument is provided, it will compute row wise or column sum else normal summation of all the element of the array.

'''print(np.sqrt(arr1))

# np.sum(x, axis)

a = arr1.reshape(2, 2)
print(a)

b = np.sum(a, axis = 0) # the same column elements are added
print(b)
print(np.sum(a))

b = np.sum(a, axis = 1) # the same row elements are added
print(b)'''

'''a = arr1.reshape(2,2)
print(a.T)'''

# Performing Arithmatic operations on matrix using loops

'''arr1 = [[1, 2], [3, 4]]
arr2 = [[5, 6], [7, 8]]

row = len(arr1)
col = len(arr1[1])

# Creating a matrix of size 2 x 2 of 0, for adding the sum into it
new = [[0 for i in range(row)] for j in range(col)]

for i in range(row):
    for j in range(col):
        new[i][j] = arr1[i][j] + arr2[i][j]

print(np.array(new).reshape(2, 2))'''

'''mat = []
for row in range(3):
    a = []
    for col in range(3):
        a.append(col)

    mat.append(a)

for i in range(3):
    for j in range(3):
        print(mat[i][j], end = " ")

    print()'''

# Numpy has matlib library which stands for matrix library.

'''import numpy as np
import numpy.matlib

# The parameter "order" is used to speciy how the elements of a matirix are stored in the memory. The order "C" - row based order. The order "F" - column based order.

# It helps us to print random numbers.
obj = np.matlib.empty((2, 3), dtype = int, order = "F")
print(obj)

# np.matlib.zeros() - returns a matrix filled with Zeros.
mat = np.matlib.zeros(shape = (1, 5), dtype = int)
print(mat)

# np.matlib.ones(shape, dtype) -> returns a matrix filled with 1
amt = np.matlib.ones(shape = (2, 3), dtype = int)
print(amt)

# To print a matrix having 1 on diagonal and others as Zero
# matlib.eye() -> 3 parameters -> n = row, M = col, dtype and order are optional.
mat = np.matlib.eye(n = 5, M = 5, dtype = int, order = "F")
print(mat)

mat = np.matlib.eye(n = 5, M = 5)
print(mat.reshape((1, 25))) 

# np.matlib.identify(n, dtype - optional) -> prints desired matrix of n * n order

mat = np.matlib.identity(n = 4, dtype = int)
print(mat)'''

# Add two matrices with the help of nested loops

# importing numpy library
'''import numpy as np

# Creating Two arrays using numpy

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])

# Creating a copy of the arr1, so that whenever required, we can again use it for later.

new_arr = np.copy(arr1)
# finding the shape of the array

arr_shape = np.shape(arr1)

# Creating empty matrix of size "arr_shape"

new = np.zeros((arr_shape))

# Finding the sum of arr1 and arr2 -

for row in range(len(arr1)):
    for col in range(len(arr1[0])):
        new[row][col] = arr1[row][col] + arr2[row][col]

# printing the final sum of arr1 and arr2
print(new) 

# Finding the in-place addition of both arrays.

for row in range(len(arr1)):
    for col in range(len(arr1[0])):
        arr1[row][col] = arr1[row][col] + arr2[row][col]

print(arr1)
# printing the Original Array 1
print(new_arr)'''

# Using List Comprehension for Adding Two matrices

import numpy as np

'''arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])

T = arr1'''

# Provide 'int()' else the output will be like -

# [np.int64(8), np.int64(10), np.int64(12)]
# [np.int64(14), np.int64(16), np.int64(18)]

'''new = [[int(arr1[i][j] + arr2[i][j]) for j in range(len(arr1[0]))] for i in range(len(arr1))]

for i in new:
    print(i)'''

# Transposing a matrix 
# np.transpose(arr1), print(arr1.T)
'''print(np.transpose(arr1))
print(arr1.T)'''

# Uisng zip
''' # Explaination -> The zip method we see generally is - zip(a, b). Similarly, here afiter unpacking the 2D array, it takes each elements from the each array and zip them together. Therefore -> [1,2,3] [4,5,6] is zipped into [1,4] [2,5] [3,6]

arr1 = [[1, 2, 3], [4, 5, 6]]

result = zip(*arr1) # The astrik here, unpacks the list into its elements. Therefore the arr1 becomes -> [1,2,3], [4,5,6]

for row in result:
     # We know that we have array created by numpy. Therefore, the elements are always in the form - "numpy.int64(element at 0)". Therefore directly printing the row gives result as shown below. Also it is tuple as it is unpacked from zip function. So direct conversion into integer won't work. The first we need to change it into list and then the element in integer.

    

    print([int(x) for x in row]) # Now if we convert the elements of row into integer then we can get normal answer or desired one according to our need. or simply it instructs to do the following -> x, which is in row, convert it into integer and then whole into a list.
    

# Now this does produce any output because the zip function is an iterator. and an Iterator when consumed, cannot be used.
for j in result:
    print([int(x) for x in j])

# Other Iterators that cannot be used again ones consumed -
# enumeratre(), map(), zip(), reversed(), itertools.chains(), itertools.cycle() -> it cycles through the range indefinitely. , itertools.permutations() and some others.'''

# Cycle 
'''import itertools
from itertools import cycle

result = itertools.cycle([1,2])

for i in result:
    print(i)'''


# Creating Matrix of N * N -

'''row = 3
col = 3

new = []

for i in range(row):
    a = []
    for j in range(col):
        a.append(j)
    new.append(a)

for i in range(row):
    for j in range(col):
        print(j, end = " ")
    print()'''

# Get the Kth column of the matrix

'''import numpy as np

arr = [[1,2,3], [4,5,6], [7,8,9]]

arr1 = np.array(arr)'''


'''for x in np.nditer(arr1, flags = ["buffered"], op_dtypes = ["S"]):
    pass

for x in  np.nditer(arr1[:, :1:2]): # [total length : index from which slicing to do (can be :x or x:) : step size]
    print(x, end = " ")'''
    

'''new = [a[2] for a in arr] # a[2] is the subarray of array "arr" and 2 is the index of element.
print(new)

# we can also use zip - 
res = list(zip(*arr))[2]
#or 
new = zip(*arr)
print(list(list(new)[2]))

# using numpy 
new = np.array(arr1[:, 1])
print(new)

# inner -
a = np.array([[1,2], [4,5]])
b = np.array([[7,8], [10,11]])

print(np.array([a,b])) # a[0] * a[0] and a[0] * a[1]
# then a[1] * a[0] and a[1] * a[0]

print(np.inner(a,b))'''
 

'''import numpy as np

arr = np.array([[1,2,3], 
                [4,5,6]])

for i in np.nditer(arr, order = "C", flags = ["external_loop"]):
    print(i)'''


# Random Numbers in Numpy
# It does  not mean a different number every time. Random means something which can't be predicted Logically.

'''from numpy import random

# random.randint(n)
x = random.randint(20)
print(x)


y = random.randint(100)
print(y)

new = np.array([x,y])
print(new)'''

# To generate Float Number

# random.rand(n) -> The paramenter passed produces n elements which always lies between 0 and 1.
'''from numpy import random

x = random.rand()
print(x)

y = random.rand(3)
print(y)

print(np.array([random.rand(), random.rand()]))
print(np.array([random.rand(2), random.rand(2)]))'''


# Generate Random Array
'''from numpy import random'''

'''x = random.rand(4)
print(x)
'''

'''print(random.rand(4).dtype)'''
# we can also use randint() method. We can pass parameter "size" to define the shape of the array.
'''x = random.randint(5, size = (2,2))
print(x)

y = random.randint(200, size = (2,2))
print(y)


z = np.add(x, y)
print(z)

zz = np.array([x, y]) # creates a 3D array
print(zz)

z = np.unique(zz) # finding unique elements
print(z)

z = np.intersect1d(x, y) # finding intersection 
print(z)

z = np.setdiff1d(x, y, assume_unique = True)
print(z)


# finding symmetric difference -> applies only on 1D arrays.
z = np.setxor1d([12,3,4], [12, 3, 5], assume_unique = True)
print(z)'''

# we can print 1D array of particular size -


'''from numpy import random

x = random.randint(5, size = 4)
print("The array is : ", x)

random_number = random.randint(10)
random_number2 = random.rand(3)

print(random_number2.dtype)
print(random_number)

if random_number < 4:
    print(random_number2.dtype)
    print(random_number2)


# generating random array with random.rand()
# passed 2 parameters

a = random.rand(3, 5)
# it will print 3 rows containing 5 elements
print(a)'''


# Generate Random Number from Array -
# choice()

'''from numpy import random

x = random.choice([1,2,3,4])

print(x)

# creates array of choice of given size
x = random.choice([2,4,6,8], size = (2, 3))
print(x)
'''


# Random Distribution 
# It is a set of all possible values and how often each values occurs.
# They follow a certain Probability Destiny Function -> a function that describes a continous probability of all values in an array.

'''from numpy import random

arr = random.choice([3, 5, 7, 9], p = [0.1, 0.3, 0.5, 0.1], size = (3,3)) 
# The probability for each number to occur in the array is set. The probability of those numbers can be atmost equal to correspoonding number of another array "p", in the array "arr" to occur.

print(arr)

# Note -> The sum of all probability should be = 1.

arr = random.choice([7, 5, 8, 1], p = [0.5, 0.5, 0, 0], size = (3, 3))
print(arr)'''


# Numpy Permutations 
import numpy
from numpy import random

arr = np.array([5, 3, 7, 0])

random.shuffle(arr)

print(arr)

arr = np.array([5,2,5,6])
random.permutation(arr)
print(arr)




