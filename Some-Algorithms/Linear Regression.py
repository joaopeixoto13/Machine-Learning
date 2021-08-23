# Predict Employee Salary based on years experience

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Import the Dataset
salary_data = pd.read_csv("Salary_Data.csv")
X = salary_data.iloc[:, :-1].values             # Pick 1 column
y = salary_data.iloc[:, 1].values               # Pick 2 column

# Draw a graphic bar
# sns.barplot(x="YearsExperience", y="Salary", data=salary_data)
# plt.show()


# Split the variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Create the Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Show the score
print(f'score: {round(model.score(X_test, y_test) * 100, 2)}%')

# Predict the results
y_pred = model.predict(X_test)

for i, k in enumerate(y_pred):
    error = abs((y_test[i] - k) / y_test[i])
    error = round((error * 100), 2)
    print(f'Actual: {y_test[i]}\t, Prediction: {int(k)}, \tError: {error}%')

# Visualize the test results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()