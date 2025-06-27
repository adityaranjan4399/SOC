# 1. Python Basics
src/python_basics/variables.py


# Variables and Data Types
name = "Alice"
age = 25
print(name, age)
src/python_basics/loo

# Loops
for i in range(5):
    print(i)
src/python_basics/functions.py


# Functions
def greet(name):
    return f"Hello, {name}!"

print(greet("Bob"))
src/python_basics/data_structures.py


# Data Structures
fruits = ["apple", "banana", "cherry"]
student = {"name": "Alice", "age": 25}
print(fruits, student)
src/python_basics/file_handling.py


# File Handling
with open("example.txt", "w") as f:
    f.write("Hello, world!")
src/python_basics/oop.py


# Object-Oriented Programming
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hi, my name is {self.name}")

p = Person("Alice", 25)
p.greet()
2. NumPy Module
src/numpy_examples/arrays.py


import numpy as np

# Array Creation
a = np.array([1, 2, 3])
print(a)
src/numpy_examples/broadcasting.py


import numpy as np

# Broadcasting
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([1, 2, 3])
print(x + y)
src/numpy_examples/math_operations.py


import numpy as np

# Mathematical Operations
a = np.array([1, 2, 3])
print(np.sum(a))
print(np.mean(a))
print(np.std(a))
3. Pandas Module
src/pandas_examples/dataframes.py


import pandas as pd

# DataFrame Creation
data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)
print(df)
src/pandas_examples/cleaning.py


import pandas as pd

# Data Cleaning
df = pd.DataFrame({'Name': ['Alice', None], 'Age': [25, 30]})
df.dropna(inplace=True)
print(df)
src/pandas_examples/filtering.py


import pandas as pd

# Data Filtering
df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
print(df[df['Age'] > 25])
src/pandas_examples/merging.py


import pandas as pd

# Merging DataFrames
df1 = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
df2 = pd.DataFrame({'Name': ['Alice', 'Charlie'], 'Score': [90, 85]})
merged = pd.merge(df1, df2, on='Name', how='left')
print(merged)
4. Matplotlib Module
src/matplotlib_examples/plots.py


import matplotlib.pyplot as plt

# Line Plot
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot')
plt.show()
src/matplotlib_examples/scatter_plot.py

python
import matplotlib.pyplot as plt

# Scatter Plot
plt.scatter(x, y)
plt.show()
src/matplotlib_examples/histogra

import tlib.pyplot as plt

# Histogram
data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
plt.hist(data, bins=5)
plt.show()
5. Neural Network Algorithms
src/neural_networks/simple_nn.py


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example Training Loop
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
w1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
w2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

for i in range(1000):
    # Forward Pass
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    # Backpropagation and Weight Updates (simplified)
    error = y - a2
    dZ2 = error * (a2 * (1 - a2))
    dW2 = np.dot(a1.T, dZ2)
    dB2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, w2.T)
    dZ1 = dA1 * (a1 * (1 - a1))
    dW1 = np.dot(X.T, dZ1)
    dB1 = np.sum(dZ1, axis=0, keepdims=True)
    # Update weights
    w1 += dW1
    b1 += dB1
    w2 += dW2
    b2 += dB2
For Siamese neural networks, refer to the PDF if you implement it, but a basic example is included below for reference.

src/neural_networks/siamese_nn.py


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example of a Siamese Network (simplified)
def siamese_network(input1, input2, w1, b1):
    hidden1 = sigmoid(np.dot(input1, w1) + b1)
    hidden2 = sigmoid(np.dot(input2, w1) + b1)
    distance = np.sum((hidden1 - hidden2)**2)
    return distance

# Example usage
w1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
input1 = np.array([[0.5, 0.6]])
input2 = np.array([[0.4, 0.7]])
print(siamese_network(input1, input2, w1, b1))
6. Assignments
assignments/python_basics/assignment1.py

python
# Assignment: Write a function that returns the sum of a list of numbers
def sum_list(numbers):
    return sum(numbers)

print(sum_list([1, 2, 3, 4]))
assignments/numpy/assignment1.py


import numpy as np

# Assignment: Create a 2x2 array and compute the mean
arr = np.array([[1, 2], [3, 4]])
print(np.mean(arr))
assignments/pandas/assignment1.py

python
import pandas as pd

# Assignment: Load a CSV, drop missing values, and save cleaned data
df = pd.read_csv('data.csv')
df.dropna(inplace=True)
df.to_csv('cleaned_data.csv', index=False)
assignments/matplotlib/assignment1.py


import matplotlib.pyplot as plt

# Assignment: Plot a bar chart
plt.bar(['A', 'B', 'C'], [3, 5, 2])
plt.show()
assignments/neural_networks/assignment1.py

python
import numpy as np

# Assignment: Implement a simple neural network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
w1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
w2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

for i in range(1000):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    error = y - a2
    dZ2 = error * (a2 * (1 - a2))
    dW2 = np.dot(a1.T, dZ2)
    dB2 = np.sum(dZ2, axis=0, keepdims=True)
    w2 += dW2
    b2 += dB2
7. Resources
resources/youtube_links.md

text
- **Python for Beginners – Complete Playlist:** https://youtube.com/playlist?list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&feature=shared
- **NumPy Video Tutorial (Hindi):** https://www.youtube.com/watch?v=awP79Yb3NaU
- **NumPy Video Tutorial (English):** https://www.youtube.com/watch?v=QUT1VHiLmmI
- **Pandas Video Tutorial (Hindi):** https://www.youtube.com/watch?v=JjuLJ3Sb_9U&list=PLjVLYmrlmjGdEE2jFpL71LsVH5QjDP5s4&index=2
- **Matplotlib Video Tutorial (Hindi):** https://www.youtube.com/watch?v=9GvnrQv138s&list=PLjVLYmrlmjGcC0B_FP3bkJ-JIPkV5GuZR
