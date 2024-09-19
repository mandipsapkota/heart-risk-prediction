# Heart Disease Prediction

## Overview

This project builds a heart disease prediction model using Logistic Regression. The model predicts whether a patient has heart disease based on various health features.

## Getting Started

### Requirements

Install the necessary Python libraries:

```bash
pip install numpy pandas scikit-learn
```

### Usage

1. **Import Libraries**

   ```python
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   ```

2. **Load and Inspect Data**

   ```python
   heart_data = pd.read_csv("/content/heart.csv") # This is taken from colab so I dont have dataset here , make sure you customize link
   print(heart_data.head())
   ```

3. **Check Data**

   - View the shape and info of the dataset:

     ```python
     print(heart_data.shape)
     print(heart_data.info())
     ```

   - Check for missing values:

     ```python
     print(heart_data.isnull().sum())
     ```

4. **Prepare Data**

   ```python
   X = heart_data.drop(columns="target")
   Y = heart_data['target']
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
   ```

5. **Train Model**

   ```python
   model = LogisticRegression()
   model.fit(X_train, Y_train)
   ```

6. **Evaluate Model**

   ```python
   X_train_prediction = model.predict(X_train)
   training_accuracy = accuracy_score(Y_train, X_train_prediction)
   print(f'Accuracy on training data: {training_accuracy}')
   
   X_test_prediction = model.predict(X_test)
   test_accuracy = accuracy_score(Y_test, X_test_prediction)
   print(f'Accuracy on test data: {test_accuracy}')
   ```

7. **Make Predictions**

   ```python
   input_data = (43, 1, 0, 120, 177, 0, 0, 120, 1, 2.5, 1, 0, 3)
   input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
   prediction = model.predict(input_data_as_numpy_array)
   
   if prediction[0] == 0:
       print("Heart safe")
   else:
       print("Heart disease detected")
   ```

## Conclusion

The model predicts heart disease based on patient data with good accuracy. This project demonstrates a basic application of Logistic Regression for binary classification.
