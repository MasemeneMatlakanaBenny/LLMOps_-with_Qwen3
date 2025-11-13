## test if the model has been staged to the production phase successfully or not:
from configurations_mlflow import set_mlflow_exp,set_mlflow_host
from configurations_mlflow import load_tokenizer_name,test_model_versioning


## set the mlflow experiment and host within the mlflow workflow:
set_mlflow_exp()
set_mlflow_host()


## create the model's variables:
tokenizer_name=load_tokenizer_name()
tokenizer_stage="production"

test_prod_tokenizer=test_model_versioning(name=tokenizer_name,stage=tokenizer_stage)

print(test_prod_tokenizer)
