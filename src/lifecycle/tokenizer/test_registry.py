## test if the model has been staged to the production phase successfully or not:
from configurations_mlflow import set_mlflow_exp,set_mlflow_host
from configurations_mlflow import load_tokenizer_name,test_model_registry


## set the mlflow experiment and host within the mlflow workflow:
set_mlflow_exp()
set_mlflow_host()


## create the model's variables:
tokenizer_name=load_tokenizer_name()
tokenizer_version="1"

test_reg_tokenizer=test_model_registry(name=tokenizer_name,version=tokenizer_version)

print(test_reg_tokenizer)
