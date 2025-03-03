"""
Logging is the process of saving metrics, parameters and artifacts to 
MLflow Tracking for an active run.

To log a single metric we can use the method log_metric.
For multiple metrics we can pass a dictionary to the log_metric.

To log parameters we can use the log_param method.

To log artifact we can use the log_artifact() passing a single file or
we can use log_artifacts() pointing to a directory containing multiple files.

Use the start_run() function to make a run active. Then we train our model.
Once the training is completed, we can log results to MLflow Tracking.

Summary log methods:
    - log_metric /log_metric(dict)
    - log_parameters
    - log_artifact / log_artifacts(file_path)

We will use a subsample of the the Unicorn dataset to do the exercise.
"""

import os
import mlflow
from mlflow.exceptions import MlflowException
from sklearn.linear_model import LinearRegression
import pandas as pd

EXPERIMENT_NAME = "exercise"

# Create an experiment
try:
    mlflow.create_experiment(EXPERIMENT_NAME)
except MlflowException as err:
    print(f"Experiment already existed [{EXPERIMENT_NAME}]")

# Set the current experiment
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

# Start a new run
linear_regression = LinearRegression()

exercise_path = os.getcwd() + "\\mlflow\\ex_04"
# Load train, test data
X_train = pd.read_csv(exercise_path + "\\X_train.csv")
y_train = pd.read_csv(exercise_path + "\\y_train.csv")["Profit"]


X_test = pd.read_csv(exercise_path + "\\/X_test.csv")
y_test = pd.read_csv(exercise_path + "\\/y_test.csv")["Profit"]

# Start a new run before training the model
run = mlflow.start_run()

# Train the model
linear_regression.fit(X=X_train, y=y_train)

# Check the score of the model
model_score = linear_regression.score(X=X_test, y=y_test)

# Now we can log the results to mlflow
mlflow.log_metric("r_squared", model_score)

# Not parameter were used so I'll leave this commented for now
# mlflow.log_params()

# Save as an artifact the code used
mlflow.log_artifact(exercise_path + "\\ex.py")

mlflow.end_run()
