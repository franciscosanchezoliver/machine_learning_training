
import mlflow
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime


# Select a SQL database to store Mlflow necessary data
MLFLOW_DATABASE = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_DATABASE)

# Searching experiments using MLflow client API
experiments = client.search_experiments()

print(f"Experiments stored in mlflow: {len(experiments)}")

print("\nExperiments")
print("-----------")

for experiment in experiments: 
    print(f"""
    Name: {experiment.name}
    Experiment id: {experiment.experiment_id}
    Artifact location: {experiment.artifact_location}
    Creation: {datetime.utcfromtimestamp(experiment.creation_time)
                       .strftime("%d-%m-%Y %H:%M:%S")}
    Last Update: {datetime.utcfromtimestamp(experiment.last_update_time)
                          .strftime("%d-%m-%Y %H:%M:%S")}
    Lifecycle stage: {experiment.lifecycle_stage}
    """)



print("end")




