import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import json
import os

# Path to the parameters file
params_file = 'best_params.json'

# Load the dataset and clean, remove 0 for OEE, convert to categorical data
df = pd.read_csv(r'c:\Users\JFOGARTY\Downloads\train_set.csv')
df = df[df['OEE'] != 0]

X = df.iloc[:, [i for i in range(df.shape[1]) if i != 2]]
y = df['OEE']
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Check if best parameters file exists
if not os.path.exists(params_file):
    # Define the parameter grid for RandomizedSearchCV
    param_distributions = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(random_state=23)

    # Perform Randomized Search CV
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, 
                                   n_iter=15, cv=3, verbose=2, random_state=21, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    # Save the best parameters
    best_params = rf_random.best_params_
    with open(params_file, 'w') as file:
        json.dump(best_params, file)
else:
    # Load the best parameters
    with open(params_file, 'r') as file:
        best_params = json.load(file)

print("Best Parameters:", best_params)

# Initialize the model with the best parameters
best_rf = RandomForestRegressor(**best_params, random_state=21)


# Train the model with the best parameters
best_rf.fit(X_train, y_train)

# Predict on the test set
final_y_pred = best_rf.predict(X_test)

# Evaluate performance
mae = metrics.mean_absolute_error(y_test, final_y_pred)
mse = metrics.mean_squared_error(y_test, final_y_pred)
rmse = np.sqrt(mse)
mape = metrics.mean_absolute_percentage_error(y_test, final_y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)

