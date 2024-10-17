# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Preprocess Data
2. Split the Data
3. Train the Model
4. Evaluate and Predict

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Preethi S
RegisterNumber:  212223230157
*/
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()

![image](https://github.com/user-attachments/assets/df6bf877-1829-4c05-ad9b-27a4d44d4d6f)
```
data.info()
```
![image](https://github.com/user-attachments/assets/f56d7d58-5c54-497b-81a0-62e212a64232)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/3417e7f9-9324-4282-923c-daac2d2da302)

```
data["left"].value_counts()
```

![image](https://github.com/user-attachments/assets/2abdffe0-176e-4298-b216-44cfdafb3c6c)

```
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```

![image](https://github.com/user-attachments/assets/2458940f-f614-42a7-9b6f-df617f7e7b7c)

```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data[["left"]]
y.head()
```
![image](https://github.com/user-attachments/assets/edf8fbd5-7aa2-4beb-b819-876b02f2fa39)

```
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
```
![image](https://github.com/user-attachments/assets/ffbaaf56-03cd-45fb-a402-d49202e39e7e)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

