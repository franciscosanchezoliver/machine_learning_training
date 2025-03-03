"""
Example:
1- creating an experiment 
2- Set a tag to the experiment
3- Set a experiment
"""

import mlflow

# Create the experiment with a name
# Enclose this creation in a try except in case the experiment was already
# created in a previously execution
try:
    mlflow.create_experiment(name="ex_1")
except Exception as err:
    print(err)

# Set a tag to the experiment
mlflow.set_experiment_tag("sklearn", "lr")

# Set the current experiment
mlflow.set_experiment("ex_1")
