# kkbox-churn-prediction
Kaggle Churn Prediction project

<h2>Instructions</h2>

<strong>1.</strong> Download data.npz and load_data.py

<strong>2.</strong> Run load_data.py (or use it as a function in a file that requires the training data to be loaded)

<h2>Desription of each file</h2>

<strong>filereader.py</strong> Reads data from the main training set and stores it as numpy arrays in data.npz after processing the data into the numpy arrays X_train (features) and Y_train (output: churn or no churn)

<strong>data.npz</strong> Stores the numpy arrays X_train and Y_train

<strong>load_data.py</strong> Extracts X_train and Y_train from data.npz
