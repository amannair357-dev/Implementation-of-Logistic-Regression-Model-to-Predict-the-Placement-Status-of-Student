# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Start and import the required libraries and student dataset.
2.Split the dataset into input features X and placement status y.
3.Create a Logistic Regression model and train it using the training data.
4.Predict the placement status using the trained model.
5.Evaluate the model using accuracy score and display the result.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Aman Nair
RegisterNumber:  212225240008
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Load data
data = pd.read_csv("Placement_Data (1).csv")

#Remove salary because it is not known before placement
data = data.drop("salary", axis=1)

#Convert categorical data into numerical data
data = pd.get_dummies(data, drop_first=True)

#Features and target
X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]

#Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Accuracy
print("Accuracy:", model.score(X_test, y_test))

#Predict a new student
prediction = model.predict(X_test)

print("Predicted Placement Status:")
print(prediction)
```
## Output:

<img width="675" height="128" alt="image" src="https://github.com/user-attachments/assets/e93a4b5d-ccd5-487a-a5e6-189294c259bf" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
