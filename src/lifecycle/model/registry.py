
## lets register the model on mlflow here:

### start first by import the necessary libraries first:
import mlflow
from configurations_mlflow import set_mlflow_host,set_mlflow_exp,load_model_name
from configurations import load_model

## now set the mlflow_host and mlflow created experiment within the workflow:

set_mlflow_host()
set_mlflow_exp()

## create the model + model name variables:
model=load_model()
model_name=load_model_name()

## register the model on the mlflow created experiment:
run_name="model_registry"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.transformers.log_model(transformers_model=model,registered_model_name=model_name)
