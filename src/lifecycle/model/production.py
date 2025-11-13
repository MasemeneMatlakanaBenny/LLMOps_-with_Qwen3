## lets register the model on mlflow here:

### start first by import the necessary libraries first:
import mlflow
from configurations_mlflow import set_mlflow_host,set_mlflow_exp
from configurations_mlflow import load_model_name,mlflow_client

## now set the mlflow_host and mlflow created experiment within the workflow:

set_mlflow_host()
set_mlflow_exp()

## create the model + model name variables:
model_name=load_model_name()
model_version="1"

## create the mlflow client:
client=mlflow_client()

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="production"
)
