**Linear Regression model**
Difference between software engineering Program abd ML program

In software program we provide the input and add the logical in the program so that we can get the desired output , However in the ML program we are providing the input(feature) and Target(label) to get the desired output where the ML will apply the algorithm to predict the output

<img width="2188" height="594" alt="image" src="https://github.com/user-attachments/assets/d509709f-bb3c-41c1-b752-45762a4b9b7c" />

MLflow looks as below which elaborates how the ML datasets are analysed and send the data to the ML and make use of algorthms to get the predicted output .

There are multiple algorithms in the ML , In the below example we are showing Linear Regression algoirthms 

<img width="2128" height="520" alt="image" src="https://github.com/user-attachments/assets/617695d2-7ee3-4d2f-8d95-341e652100f8" />

Library used to use the algoirthm to feed the training data is used based on the category 
  https://scikit-learn.org/stable/

Linear Regression is a superviser learning  : IT takes the argument as X and Y 

To analyse the data between the 2 entities like input and output we are usimg the library as matplotlib https://matplotlib.org/

We need to split the data for training and testing , In order to do this task the scikit library have a function https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

**Supervised Learning** --> In the Supervised Learning we are providing the Features (inputs) and the target for the ML model ton predict the output 


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
df=pd.read_csv('data copy.csv')
'''
print(df.head(15))
print(df.info())
print(df.shape)
print(df.columns) # This will give us the column names of the datasets 
print(df.describe())
print(df.duplicated().sum())
print(plt.figure(figsize=(10,6)))
plt.scatter(df['bedrooms'],df['price'])
plt.xlabel('No of bedrooms')
plt.ylabel('Price')
plt.show()'''
X = df[['bedrooms']]
y = df['price'] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
model=LinearRegression()
print(model.fit(X_train,y_train))

joblib is the library used to save the model in the file format
