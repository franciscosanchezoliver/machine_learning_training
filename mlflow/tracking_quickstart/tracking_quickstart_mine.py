import mlflow
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Selecting the tracking URI
# --------------------------
# For this case I've decided to store the data in a SQL database previously
# created:
#   Note: 
#        Before trying to access the database, it must previously created. 
#        Use the following command to create the database:
#        mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_DATABASE = "sqlite:///mlflow.db" 
mlflow.set_tracking_uri(MLFLOW_DATABASE)

# Load data and train a simple model.
# -----------------------------------
# Just to try mlflow we'll load data from the iris dataset.
iris_data = datasets.load_iris()



X = iris_data.data
y = iris_data.target

# X, y = sklearn.datasets.load_iris(return_X_y=True)

print(f"""{iris_data.DESCR}

Feature Names
-------------
{iris_data.feature_names}

Target Name
-----------
{iris_data.target_names}

Data Size
---------
Rows: {X.shape[0]}
""")


# Split data into train and test sets
# -----------------------------------
TEST_SIZE = 0.2
RANDOM_STATE= 42

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=TEST_SIZE, 
                                                    random_state=RANDOM_STATE)

print(f"""Train size
----------
{X_train.shape[0]}

Test size
---------
{X_test.shape[0]}
""")

# Use a Logistic Regression model for the classication
LOGISTIC_REGRESSION_HYPERPARAMETERS_RANDOM_STATE = 8888
logistic_regression_hyperparameters = {
    # Limited-memory BFGS
    # Is an optimization algorithm in the family of quasi-Newton methods that approximates 
    # the Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS) using a limited amount 
    # of computer memory.It is a popular algorithm for parameter estimation in 
    # machine learning
    "solver": "lbfgs", 
    "max_iter": 1000, 
    "multi_class": "auto", 
    "random_state": LOGISTIC_REGRESSION_HYPERPARAMETERS_RANDOM_STATE
}

logistic_regression = LogisticRegression(**logistic_regression_hyperparameters)


print(f"""**
Model
-----
{logistic_regression}

Hyperparameters
---------------
{LOGISTIC_REGRESSION_HYPERPARAMETERS_RANDOM_STATE}
""")

# Train the model with the training data
print("Training model... ")
logistic_regression.fit(X_train, y_train)
print("Model trained")

print("Using trained model to predict test data")
y_pred = logistic_regression.predict(X_test)

print(f"""Predictions on test set
-----------------------""")
for i, test_row in enumerate(X_test):
    print(f"{test_row} -> {y_pred[i]}")

print(f"""Predictions Correct/Incorrect
-----------------------------
Real - Predicted | Equality""")

total_of_equals = 0
total_of_differents = 0 
for i, real_test_value in enumerate(y_test):
    predicted_value = y_pred[i]
    predicted_and_real_are_equals = real_test_value == predicted_value
    print(f"""{real_test_value} - {predicted_value} | {predicted_and_real_are_equals}""")

    if predicted_and_real_are_equals:
        total_of_equals += 1
    else:
        total_of_differents += 1

accuracy = accuracy_score(y_pred, y_test)
print(f"""
  Correctly predicted: {total_of_equals}
Incorrectly predicted: {total_of_differents}

Accuracy: {accuracy}
""")

# Experiment
# ----------
# In order to group runs of a particular idea, we can define an "Experiment"
# that will group runs together, defining an unique name that is relevant 
# to what we're working on.
experiment_name = "Logistic Regression idea"
print(f"Creating a new Experiment: {experiment_name}")
mlflow.set_experiment(experiment_name=experiment_name)

print("Starting a new run of the experiment...")

with mlflow.start_run():

    print("Registering Hyperparameter in MLflow")
    print(logistic_regression_hyperparameters)
    mlflow.log_params(logistic_regression_hyperparameters)

    # Create a tag to know later what this run was for
    tag = "Training info"
    useful_description = "Basic Logistic Regression model for iris data"
    print(f"""
        Creating the following tag for the experiment:
            - {tag} : {useful_description}
          """)
    mlflow.set_tag(tag, useful_description)

    # Infer the model signature
    signature = infer_signature(X_train, logistic_regression.predict(X_train))
    print(f"Signature of the model: {signature}")

    print("Registering model in MLflow")
    model_info = mlflow.sklearn.log_model(
        sk_model = logistic_regression, 
        artifact_path = "iris_model", 
        signature = signature, 
        input_example = X_train, 
        registered_model_name = "Logistic Regression idea for Iris"
    )


print("end")