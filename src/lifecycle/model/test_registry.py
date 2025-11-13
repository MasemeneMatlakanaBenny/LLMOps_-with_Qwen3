## test if the model has been staged to the production phase successfully or not:
from configurations_mlflow import set_mlflow_exp,set_mlflow_host
from configurations_mlflow import load_model_name,test_model_registry


## set the mlflow experiment and host within the mlflow workflow:
set_mlflow_exp()
set_mlflow_host()


## create the model's variables:
model_name=load_model_name()
model_version="1"

test_reg_model=test_model_registry(name=model_name,version=model_version)

print(test_reg_model)
