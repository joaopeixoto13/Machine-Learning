# Use to solve CLASSIFICATION problems !!!

# PROBLEM:
# Predict if a person will buy an SUV based on their Age and Estimated Salary

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# Import the Dataset
social_network = pd.read_csv("SocialNetworkAds.csv")

# Extract the Independent variables
X = social_network.iloc[:, [2, 3]].values

# Extract the Dependent variables
y = social_network.iloc[:, 4].values

# Visualize the dataset
# sns.heatmap(social_network.corr())
# plt.show()

# Split the data into Training and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling - scale for better results !!!
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Train the dataset
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# Predicting the Test set Results
y_pred = model.predict(X_test)

for i, k in enumerate(y_pred):
    print(f'Actual: {y_test[i]}, Predicted: {k}')

# Visualize the Test set Results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                     np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
plt.contour(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate (np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color = ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logist Regression')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()


# Evaluating the Model using Confusion Matrix
from sklearn.metrics import confusion_matrix
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

plt.show()




