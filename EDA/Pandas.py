# Pandas is a python library which is used for working on data sets.

import pandas as pd

data = {
    "Cars": ['BMW', 'Audi', 'Mercedes'],
    "Count" : [2, 1, 2]
}

new = pd.DataFrame(data)
print(new)


print()
# Series - like a column in a Table which hold 1D array
print("Series -")

a = [1,5,7,8]
new = pd.Series(a)
print(new)

print()
# Labels - are generally the Columns which holds the values...could be index, any other values etc. The default is index of the values specified. Else can be provided by user

'''import pandas as pd

data = {
    "Cars": ['BMW', 'Audi', 'Mercedes'],
    "Count" : [2, 1, 2]
}


new = pd.DataFrame(data, index = [1, 2, 3])
print(new)

# any particular element can be accessed using Series
new = pd.Series(['BMW', 'Audi', 'Mercedes'], index = [1, 2 ,3])
print("The car at index 1 is", new[1])


print()

# The below code returns Nan values at the passed index. The reason is because when we pass Dictionary in Series, the Series method takes the key of dict as index and values as value of Series. But when an external label is passed, it try to find out keys as passed index in the dict. and as we know, [1, 2, 3] are index passed, there are no index containing 1, 2 or 3 in "data" dict, therefore the values appears to be "NaN".

new = pd.Series(data, index = [1, 2, 3])
print(new)

print()

# Using Dict in Series
new = pd.Series(data)
print(new)

print()

# we can also pass the particular key of the dict and print its values
new = pd.Series(data, index = ['Count'])
print(new)


# we can also access series element by their passed labelled values. Instead of using 1, 2, 3...n integers in index, use the labbelled tags to access them. -


print(new['Count']) # passing labels

# We can only pass numbers when we have 1D array passed in Series. 

print()

a = [2, 4, 6, 8]
new = pd.Series(a, index = [1, 2, 3, 4])
print(new[1]) '''# passing index




'''import pandas as pd

my_dict = {
    "Name" : ["Moon", "Swan", "Lion"],
    "Age" : [float('inf'), 21, 20]
}

print(my_dict)


print()
# Working with Series 
new = pd.Series(my_dict)
print(new)

print()

idx = [1, 2, 3]
new = pd.Series(my_dict)
print(new["Name"])'''




# DataFrame -> A Pandas DataFrame is a 2D Data Structure or a table with rows and columns.


'''import pandas as pd

data = {
    "Name" : ["Moon", "Swan", "Lion"],
    "Age" : [float('inf'), 21, 20]
}


df = pd.DataFrame(data)

print(df)

print()

# Locate Rows -> variable_name.loc[n], n is the specified columns we want to access. It returns Pandas Series.

print(df.loc[0])
print(df.loc[1])
print(df.loc[2])


# We can also pass a list of indexes we want to see. It is returned as DataFrame.
print(df.loc[[0, 1]])
print(df.loc[[1, 2]])
print(df.loc[[0, 2]])
print(df.loc[[0, 1, 2]])

new_df = df.loc[[1, 2]]
print(new_df)'''



'''import pandas as pd

data = {
    "Country" : ["India", "Swis", "New Zealand", "USA"],
    "Capital" : ["New Delhi", "Bern", "WEllington", "Washington DC"]
}

# Using Named Indexes
df = pd.DataFrame(data, index = ["C1", "C2", "C3", "C4"])

print(df)


#df = pd.DataFrame(data, index = [i for i in range(1, len(data["Country"]) + 1)])


# Locate Named Indexes
idx = df.loc[["C2", "C4"]]
print(idx)



# Passing Series in Dict -

data = {
    "Country" : ["India", "Swis", "New Zealand", "USA"],
    "Capital" : ["New Delhi", "Bern", "WEllington", "Washington DC"],
    "Continent" : pd.Series(["US", "Europe"], index = [4, 2]) # all the others index will be assigned NaN values.
}

new = pd.DataFrame(data, index = [1, 2, 3, 4])
print(new)
'''



# How to Load Files in DataFrame -
# CSV or Comma Separated Values Files are loaded into DataFrame

# syntax => Load FIle => data = pd.read_csv("File Location")
# syntax => Convert File => data = pd.to_csv("File Location")

'''import pandas as pd

# we can use forward slash ( / ), use raw string by putting "r" before file path, or we can use double backslash.

# the problem is that python file takes it as backspace character therefore is not able to read that file.

data = pd.read_csv(r"C:\\Users\himan\Documents\Students_details.csv")

print(data)

print()
# print columns
print(data.columns)'''




'''import pandas as pd

data = {
    "Country" : ["India", "Swis", "New Zealand", "USA"],
    "Capital" : ["New Delhi", "Bern", "Wellington", "Washington DC"],
    "Continent" : pd.Series(["Asia", "Europe", "Oceania", "US"], index = [1, 2, 3, 4]) # all the others index will be assigned NaN values.
}

new = pd.DataFrame(data, index = [1, 2, 3, 4]) # as we have created our dict and hence the default index we will get is from 0 to 3. But in Continents key, we are passing index 4 for US, which does not exist in default index. Therefore it throws error. So pass new index containing 4 and 2 to work it correctly.

print(new["Country"])
'''


'''import pandas as pd

data = {
    "Country Name" : ["USA", "Swis", "India"],
    "Temperature" : [12, 15, 30],
    "Wind_Speed" : [12, 17, 9]
}

new = pd.DataFrame(data)
print(new)

print(new['Temperature'])

# performing different mathematical operations on our data dict -
print(new['Temperature'].max())
print(new['Temperature'].mean ())

print()

# describe() -> describes all the columns of the dataset
print("Details of the Table -")
print(new.describe())

# Printing selected portion of the dataset

a = new[new.Temperature > 12 ]
print(a)

# Find the details of maximum temperature
a = new[new.Wind_Speed == new.Wind_Speed.max()]
print(a)

# Incase our Column names contains spaces, then use -> new["Name Space"].max()

print(new[new["Country Name"] == new.Wind_Speed.max()])'''






#**************************************************

'''import pandas as pd

data = pd.read_csv(r"C:\\Users\\himan\\Desktop\\Weather.csv")
print(data)'''



# Head and Tail Function
'''print(data.head(3)) # prints the dataframe upto index number 2
print(data.tail(3)) # prints the last 3 records
# Printing the shape of the DataFrame

row, col = data.shape
print(row,"x", col)


# Prints the Columns 
print(data.columns)

# To print particular Column -
a = data.Day
print(a)

# printing the temperature of the dataset
b = data.Temperature
print(b)

# We can do this in other way too -
print(data["Temperature"])
print(data["Day"])'''


# Pandas Columns are always Series Type.
#print(type(data.Day))
 

# Performing Mathematical operations on our Data -

'''maxi = data["Temperature"].max()
print("The maximum temperature is : ", maxi)

# To print all the Maximum value from the Table -
total_max = data.max()
print(total_max)

mini = data["Temperature"].min()
print(mini)

print(data.min())'''


# Printing all the Maximum and the Minimum Values from the Table -

'''new = {
    "Max" : data.max(),
    "Min" : data.min()
}

newb = pd.DataFrame(new, index = data.columns)
print(newb)'''



'''des = data.describe()
print(des)'''

# Conditionally selecting data
'''a = data[data.Temperature >= 30]
print(a)

# using Square brackets are generally more helpful in cases where our Column name contains spaces.
a = data[data["Temperature"] == data["Temperature"].max()]
print(a)
  

# print(data[data.Temperature]) -> Error because Series passed in Data perform Boolean Indexing but here no condition is provided to perfrom Bool operation.

# Printing specific columns 
a = data[["Event", "Temperature"]][data["Temperature"] == 30]
print(a)'''

'''print()
# Setting Custom Index
print(data.index)

# setting custom index
new = data.set_index("Day") # Setting Day as our Index
print("New -\n", new)

# The "new" is new DataFrame. It does not modify the original one.

# To modify the original one -> we'll pass "inplace = True"
data.set_index("Day", inplace = True)

print()
print("Modified DataFrame -")
print(data)

print()
print(data.loc["01-03-2017"])

# Resetting Index
data.reset_index(inplace = True)
print(data)

# setting index as Event
data.set_index("Event", inplace = True)
print(data)
print(data.loc["Snow"])'''





# Read and Write Files in Pandas

'''import pandas as pd

data = pd.read_csv("C:\\Users\\himan\\Desktop\\Book1.csv")

print(data)

# Sometimes we provide the name of the Table in Excel Sheet. Then in DataFrame, all the Other Columns are Assigned Unnamed. to fix it -
# use skipros = 1 ( the row in which the heading appears)



print()


# use "skiprows = 1" or "header = 1"
data = pd.read_csv("C:\\Users\\himan\\Desktop\\Book1.csv", header = 1)

#data[data.select_dtypes(include = ["number"]) < 0] = np.nan

print(data)'''




# In opposite Case if we dont have any heading in the Excel Sheet. We can still provide it 
'''import pandas as pd
data = pd.read_csv("C:\\Users\\himan\\Desktop\\Book1.csv", names = ["Hello", "WOrld", "New", "World"], skiprows = 1) # "skiprows" Skips row number 1. "names" assign new Column names to the Sheet (and merge the former Column names into the Data, if "skiprows" is not passed). If dont have Columns name Specified, add them by passing "names".

print(data)
'''



# Using \U to include a Unicode character
'''unicode_string = "Here is a grinning face emoji: \U0001F600"
print(unicode_string)'''


'''print()
# nrows = n, used to read limited number of rows.
data = pd.read_csv("C:\\Users\\himan\\Desktop\\Book1.csv", nrows = 3)

print(data)



print()
# converting n.a. into NaN
data = pd.read_csv("C:\\Users\\himan\\Desktop\\Book1.csv", na_values = ["n.a.", -1])

print(data)


print()'''
# The above function converts all the na.a values into NaN but we can see that Revenue cannot be in -1. So we need to change it too but change it with passing in "na_values" will also change -1 from EPS, where EPS can be in Negative. So we can handle it by -

# Pass a Dictionary 

'''data = pd.read_csv("C:\\Users\\himan\\Desktop\\Book1.csv", na_values = {
    "EPS" : ["n.a."],
    "Revenue" : ["n.a.", -1],
    "Price" : ["n.a."],
    "People" : ["n.a."]
})

print(data)


# Saving our last updated CSV File into our Directory -
data.to_csv("C:\\Users\\himan\\Desktop\\New.csv", index = False)
# Index is False so that in the file no new index will be added.

# printing particular columns in Excel. Updating our CSV
data.to_csv("C:\\Users\\himan\\Desktop\\New.csv", columns = ["Tickers", "People"])'''


'''import pandas as pd

data = pd.read_csv("/home/himanshu_27/Desktop/new1.csv", parse_dates = ["day"])
print(data)

print(type(data.day[0]))
data.set_index("day", inplace = True)
print(data)

new_data = data.fillna(0)
print(new_data)

'''


# Adding new COlumn and Rows in DF
import pandas as pd

# Define a dictionary containing Students data
data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Height': [5.1, 6.2, 5.1, 5.2],
        'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}

new = pd.DataFrame(data)
print(new)

new["Address"] = [1,2,3,4]

print(new)

new.insert("NEW", 5.4, "BSC", 5)
print(new)