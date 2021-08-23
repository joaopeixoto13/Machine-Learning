# Find if a person would have kyphosis after a surgery or not
# (see the 'TreeExplanation.png)

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the Dataset
kyphosis = pd.read_csv("kyphosis.csv")


# present     --> 1
# absent      --> 0
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
kyphosis['Kyphosis'] = le.fit_transform((kyphosis['Kyphosis']))
print(kyphosis.head())

# Check if is only numbers on data
# If is all '0' --> Good!
print(kyphosis.isnull().sum())

# Extract the Independent variables
X = kyphosis.drop('Kyphosis', axis=1)

# Extract the Dependent variable
y = kyphosis['Kyphosis']

# Visualize the dataset
# plt.figure(figsize=(18, 7))
# sns.countplot(x='Age', hue='Kyphosis', data=kyphosis, palette='Set1')
# plt.show()

# Split the data into Training and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the dataset
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predicting the Test set Results
y_pred = model.predict(X_test)

# Evaluating the Model using Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

TN = cm[0][0]       # Predicted No  ( TRUE )
FN = cm[1][0]       # Predicted No  ( FALSE )
FP = cm[0][1]       # Predicted Yes ( TRUE )
TP = cm[1][1]       # Predicted Yes ( FALSE )
N = TN + FN + FP + TP
print("Total:", N)
print(f'Accuracy: {round(((TN + TP) / N) * 100, 2)}%')
print(f'Misclassification Rate: {round(((FP + FN) / N) * 100, 2)}%')
