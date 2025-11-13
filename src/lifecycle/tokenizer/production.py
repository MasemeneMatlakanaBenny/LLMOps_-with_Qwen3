## lets register the model on mlflow here:

### start first by import the necessary libraries first:
import mlflow
from configurations_mlflow import set_mlflow_host,set_mlflow_exp
from configurations_mlflow import load_tokenizer_name,mlflow_client

## now set the mlflow_host and mlflow created experiment within the workflow:

set_mlflow_host()
set_mlflow_exp()

## create the model + model name variables:
tokenizer_name=load_tokenizer_name()
tokenizer_version="1"

## create the mlflow client:
client=mlflow_client()

client.transition_model_version_stage(
    name=tokenizer_name,
    version=tokenizer_version,
    stage="production"
)
