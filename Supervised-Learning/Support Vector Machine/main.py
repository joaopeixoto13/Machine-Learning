# PROBLEM:
# Classifying muffin and cupcake recipes using support vector machine!

# Dataset: Muffins have more Flour, while Cupcakes have more butter and sugar
# Image 'data.png' show the details

# -----------------------------------------------------------------------------

# Packages for analysis
import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

# Pickle package
# import pickle

# ------------------------------------------------------------------------------

# Read the '.xlsx' file and save into recipes
recipes = pd.read_csv("new.csv")

# Plot our Data
sns.lmplot(x="Flour", y="Sugar", data=recipes, hue="Type", palette="Set1", fit_reg=False, scatter_kws={"s": 70})

# Format or pre-process our data
type_label = np.where(recipes['Type']=='Muffin', 0, 1)

# Remove first column
recipe_features = recipes.columns.values[1:].tolist()
print(recipe_features)

# Pick only Flour and Sugar
ingredients = recipes[['Flour'], ['Sugar']].values
print(ingredients)

# Fit Model
# SVC - Support Vector Classification
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)

# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# Plot our Data
sns.lmplot(x="Flour", y="Sugar", data=recipes, hue="Type", palette="Set1", fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewith=2, color='black')     # Draw hyperplane
plt.plot(xx, yy_down, 'k--')                    # Draw down parallel
plt.plot(xx, yy_up, 'k--')                      # Draw upper parallel


# Function to predict Muffin or Cupcake

def muffin_or_cupcake(flour, sugar):
    if (model.predict([[flour, sugar]])) == 0:
        print("Muffin Recipe!")
    else:
        print("Cake Recipe!")


# Example
muffin_or_cupcake(50, 20)

# Plot on the graphic
plt.plot(50, 20, 'yo', markersize='9')