# kkbox-churn-prediction
Kaggle Churn Prediction project

<h2>Instructions</h2>

<strong>1.</strong> Download data.npz and load_data.py

<strong>2.</strong> Run load_data.py (or use it as a function in a file that requires the training data to be loaded)

<h2>Desription of each file</h2>

<strong>filereader.py</strong> Reads data from the main training set and stores it as numpy arrays in data.npz after processing the data into the numpy arrays X_train (features) and Y_train (output: churn or no churn)

<strong>data.npz</strong> Stores the numpy arrays X_train and Y_train

<strong>load_data.py</strong> Extracts X_train and Y_train from data.npz

<h3>Description of the numpy arrays</h3>
<strong>X_train</strong>:

Shape: (780123, 5)
Columns:
  0: num_25
  1: num_50
  2: num_75
  3: num_985
  4: num_100

<strong>Y_train</strong>:

Shape: (780123, 1)
Each entry is a 0 or 1 to represent that either a churn or no churn has occured.

