# This code shows the use of Machine Learning to recommend a suitable engineering department for the students in the faculty based on their first year results. 
# Gradient Boosting Classifier model is used here to provide more accurate result. 
# The model is trained using a dataset containing the first year results of previous students and those who has an increased GPA after the third semester. Which means only the students who has succeeded their third semester after department selection is considered.

import pandas as pd
import numpy as np

data = pd.read_csv("Dataset.csv")

# augmenting data

from sklearn.utils import shuffle

# Augment the dataset by shuffling the existing data
augmented_data = pd.concat([data, shuffle(data)], axis=0)

# Repeat the shuffling process to further increase the dataset size
augmented_data = pd.concat([augmented_data, shuffle(augmented_data)], axis=0)

# Reset the index of the augmented data
augmented_data = augmented_data.reset_index(drop=True)

# Verify the size of the augmented dataset
print("Augmented dataset size:", len(augmented_data))

data_dum = augmented_data

# map the results
category = {"A+": 8, "A": 7, "A-": 6, "B+": 5, "B": 4, "B-": 3, "C+": 2, "C": 1, "R": 0}
data_dum["Measurements"] = data_dum["Measurements"].map(category)
data_dum["Maths"] = data_dum["Maths"].map(category)
data_dum["Mechanics"] = data_dum["Mechanics"].map(category)
data_dum["POM"] = data_dum["POM"].map(category)
data_dum["Electricity"] = data_dum["Electricity"].map(category)
data_dum["Thermodynamics"] = data_dum["Thermodynamics"].map(category)
data_dum["Computer"] = data_dum["Computer"].map(category)
data_dum["Maths2"] = data_dum["Maths2"].map(category)
data_dum["Programming"] = data_dum["Programming"].map(category)
data_dum["Fluid"] = data_dum["Fluid"].map(category)
data_dum["Electronics"] = data_dum["Electronics"].map(category)
data_dum["Manufacturing"] = data_dum["Manufacturing"].map(category)
data_dum["Drawing"] = data_dum["Drawing"].map(category)

# map the Department
category = {"CE": 0, "ME": 1, "EE": 2, "CO": 3}
data_dum["Department"] = data_dum["Department"].map(category)

from sklearn.model_selection import train_test_split

x = data_dum.drop("Department", axis=1)
y = data_dum["Department"]

# GVM

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

gbm = GradientBoostingClassifier(
    random_state=0, n_estimators=109, max_depth=7, learning_rate=0.08
)
gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict(X_test)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)

# print

print(f"GBM Accuracy: {accuracy_gbm}")

import pickle

pickle.dump(gbm, open("model.pkl", "wb"))

model = pickle.load(open("model.pkl", "rb"))
print(
    model.predict([["0", "8", "1", "2", "8", "2", "1", "0", "8", "8", "0", "8", "8"]])
)
