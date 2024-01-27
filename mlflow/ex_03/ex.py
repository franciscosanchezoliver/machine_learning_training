"""
MLFlow Tracking is organized around a concept called "runs". 
A new run means model training and information about the model is logged 
to MLFlow.
Each run is placed within an experiment.

When a training run is started, the MLFLow module sets the run as "active".
When a run is active, all metrics, parameters, and artifacts will be logged 
under the current active run. 

MLFLow module will continue logging to the active run until the code exists
or the "end_run()" method is called.
"""

import mlflow
from mlflow.exceptions import MlflowException
from datetime import datetime

experiment_name = "exercise"
# Create a new experiment
try:
    mlflow.create_experiment(name=experiment_name)
except MlflowException as err:
    print("Experiment already exist")

mlflow.set_experiment(experiment_name=experiment_name)

# Start a new run
run = mlflow.start_run()

# Print run info
print(run.info)

run_info = run.info

print(
    f""" 
Info
----

  Experiment
  ----------
    Experiment id: {run_info.experiment_id}

  Run
  ---
    Run id: {run_info.run_id}
    Start time: {datetime.fromtimestamp(run_info.start_time/1000.)}
    Run status: {run_info.status}
    Life cycle stage: {run_info.lifecycle_stage}
    Run name: {run_info.run_name}
    User id: {run_info.user_id}

  Artifact
  --------
    Uri: {run_info.artifact_uri}
"""
)
