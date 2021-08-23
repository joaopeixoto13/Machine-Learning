import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
import random

# FÓRMULAS
# y = mx + c
# F = 1.8*C + 32

x = list(range(0, 10))
#y = [1.8 * F + 32 for F in x]                               # Generate y values for us
y = [1.8 * F + 32 + random.randint(-3, 3) for F in x]        # Generate y values for us with noise

print(f'X: {x}')
print(f'Y: {y}')

# Create a plot / graphic in red color
plt.plot(x, y, '-*r')
plt.title("Temperature Estimation")
# plt.show()

# Reshape the data
# Required by library                   Format for Machine Learning
x = np.array(x).reshape(-1, 1)          # [ [0] [1] [2] ]
y = np.array(y).reshape(-1, 1)


# Split our Data in 2 categories
# 1. Train
# 2. Test
# test_size=0.2 --> 20% for Test and 80% for Train
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.2)

# print(xTrain.shape)             # From 10 values, 8 for Training


# Create the Model
model = linear_model.LinearRegression()

# Apply the xTrain and yTrain to the Model - Train the Model
model.fit(xTrain, yTrain)
print(f'Coefficient: {round(model.coef_[0][0], 2)}')            # Get coefficient
print(f'Intersection: {round(model.intercept_[0], 2)}')         # Get intersection

# Receive the score of our model (with the result of our test values)
accuracy = model.score(xTest, yTest)
print(f'Accuracy: {round(accuracy * 100, 2)} %')


# Reshape and get original 'x'
x = x.reshape(1, -1)[0]
m = model.coef_[0][0]
c = model.intercept_[0]
y = [m * F + c for F in x]
plt.plot(x, y, '-*b')
plt.show()