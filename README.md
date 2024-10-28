# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries such as pandas, sklearn, and numpy. Load the employee churn dataset and inspect it to understand the features and target (e.g., churn: yes/no)
2. Handle missing values, encode categorical variables using OneHotEncoder or LabelEncoder, and scale the data if necessary. Ensure the dataset is clean and prepared for training the model by handling outliers or inconsistent data.
3. Use train_test_split() from sklearn.model_selection to split the data. Divide it into training and testing sets (e.g., 80% training, 20% testing) for model evaluation.
4. Use DecisionTreeClassifier from sklearn.tree and fit the model on the training data.
Set hyperparameters like max_depth or min_samples_split to avoid overfitting.
5. Use the trained decision tree model to predict employee churn on the test data.
Store the predictions to compare them with the actual employee churn labels in the test set.
6. Calculate performance metrics such as accuracy, precision, recall, and F1-score using classification_report() from sklearn.metrics. These metrics will help evaluate how well the model predicts employee churn.
7. Use plot_tree() from sklearn.tree to visualize the decision tree structure.
This provides insight into how the model makes decisions, helping interpret which features are most important for predicting churn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Preethi S
RegisterNumber: 212223230157

import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull.sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data[["left"]]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/
```

## Output:
# Dataset

![Screenshot 2024-09-20 141807](https://github.com/user-attachments/assets/8e1afbe5-6cab-4f34-9f67-0c5363bef40d)

 
# Data Info

![Screenshot 2024-09-20 141815](https://github.com/user-attachments/assets/78022c77-d134-4d90-82ab-35bc151aa97b)


# Sum of Null Values

![Screenshot 2024-09-20 141819](https://github.com/user-attachments/assets/c41453b2-93a5-4f5e-abbf-c8d8fe65d6a5)


# Labelling

![Screenshot 2024-09-20 141824](https://github.com/user-attachments/assets/52b8d38f-8b1b-441e-ab07-44b72c4c18a6)


# Assignment of x and y values

![Screenshot 2024-09-20 141856](https://github.com/user-attachments/assets/655b6c0f-4a5d-40b9-a8e9-7cf70a6791bd)

![Screenshot 2024-09-20 141902](https://github.com/user-attachments/assets/8272ff21-5841-4b4f-9df8-62e1f3dbea7a)


# Converting string literals to numerical values using label encoder:

![Screenshot 2024-09-20 141837](https://github.com/user-attachments/assets/032b7667-5476-4f90-9e63-937c0d33a8e9)


# Accuracy

![Screenshot 2024-09-20 141908](https://github.com/user-attachments/assets/44287cd3-cf56-4844-abe9-a5344f043ee7)


# Prediction

![Screenshot 2024-09-20 141924](https://github.com/user-attachments/assets/1a7b1554-8bfb-40f2-bebf-78a6f0a010e4)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee
