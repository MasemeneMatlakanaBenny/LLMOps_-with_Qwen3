
## lets register the model on mlflow here:

### start first by import the necessary libraries first:
import mlflow
from configurations_mlflow import set_mlflow_host,set_mlflow_exp,load_tokenizer_name
from configurations import load_tokenizer

## now set the mlflow_host and mlflow created experiment within the workflow:

set_mlflow_host()
set_mlflow_exp()

## create the model + model name variables:
tokenizer=load_tokenizer()

tokenizer_name=load_tokenizer_name

## register the model on the mlflow created experiment:
run_name="tokenizer_registry"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.pyfunc.log_model(tokenizer,registered_model_name=tokenizer_name)
