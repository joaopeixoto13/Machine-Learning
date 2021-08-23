import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model, model_selection

print("-"*30)
print("IMPORTING DATA")
print("-"*30)

# Read the .csv file and put on data
data = pd.read_csv("houses_to_rent.csv", sep=",")

# Select the columns that we gonna use
data = data[['city', 'rooms', 'bathroom', 'parking spaces', 'fire insurance', 'furniture', 'rent amount']]

# print first 5 rows of data
print(data.head())


##### PROCESS DATA #####

# Take out the 'R$'
# R$7,000 --> 7000 (remove first 2 values and the comma ',')
# map to apply all rows
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',', '')))
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:].replace(',', '')))

# furnished     --> 1
# not furnished --> 0
le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform((data['furniture']))

# print first 5 rows of data
print(data.head())


##### VERIFY IF IS ONLY NUMBERS ON DATA #####

print("-"*30)
print("CHECKING NULL DATA")
print("-"*30)

# Check if is only numbers on data
# If is all '0' --> Good!
print(data.isnull().sum())

# Throw away 'not numbers'
# data = data.dropna()
# print(data.isnull().sum())


##### SPLIT DATA #####

print("-"*30); print("SPLIT DATA"); print("-"*30)

# Drop rent amount (axis 1) because is our Output
x = np.array(data.drop(['rent amount'], axis=1))

y = np.array(data['rent amount'])

print('X', x.shape)                 # Input  --> 6080 instances, 6 attributes (except the rend amount)
print('Y', y.shape)                 # Output --> 6080 instances

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.2)

# Verify if 20% of 6080 instances are for Testing and 80% are for Training
print("xTrain:", xTrain.shape)
print("xTest:", xTest.shape)


##### TRAINING #####

print("-"*30); print(" TRAINING "); print("-"*30)
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)
accuracy = model.score(xTest, yTest)

print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print(f'Accuracy: {round(accuracy*100, 2)} %')


#####  TESTING  #####

print("-"*30); print(" MANUAL TESTING "); print("-"*30)

# Get 'x' predict values
testValues = model.predict(xTest)

error = []

for i, testValue in enumerate(testValues):
    error.append(yTest[i] - testValue)
    print(f'Actual: {yTest[i]}R$, Prediction: {int(testValue)}R$, Error: {int(error[i])}R$')
