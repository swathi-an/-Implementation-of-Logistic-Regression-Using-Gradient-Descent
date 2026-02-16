# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Algorithm: Logistic Regression using Gradient Descent

Input: 
    Training data X (features), y (labels)
    Learning rate α
    Number of iterations N

Output:
    Weights w, bias b

Steps:
1. Initialize weights w = 0 and bias b = 0
2. Repeat for N iterations:
    a. Compute linear combination:
        z = w^T * X + b
    b. Apply sigmoid function:
        y_hat = 1 / (1 + exp(-z))
    c. Compute gradients:
        dw = (1/m) * Σ (y_hat - y) * X
        db = (1/m) * Σ (y_hat - y)
    d. Update parameters:
        w = w - α * dw
        b = b - α * db
3. Return final w, b 

## Program:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Placement_Data.csv")
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()

## Output:
<img width="492" height="796" alt="image" src="https://github.com/user-attachments/assets/f2911fda-ce21-4655-b40b-b2ebdc84ec16" />
![WhatsApp Image 2026-02-03 at 10 48 02 AM](https://github.com/user-attachments/assets/c73c8eb7-9b4d-4bbb-8f38-708974519c75)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

