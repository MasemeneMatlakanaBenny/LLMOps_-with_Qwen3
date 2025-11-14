## create the functiont that we will use for mlflow workflow:

host="http://127.0.0.1:5000"

exp_name="LLMOPs_with_QWEN3"

exp_description="Designing an entire lifecycle of LLMs with Qwen3 model from huggingface"

tags={
    "project_name":"LLMopS_QWEN3",
    "team":"AI/ML Team",
    "team lead":"Masemene Matlakana Benny",
    "project_start_date":"12 November 2025",
    "mlflow.note.description":exp_description
}

model_name="qwen3_model"

tokenizer_name="qwen3_tokenizer"


## create a function for loading the tracking uri/local host/server for mlflow:
def load_host():
    return host

##create a function that will be used to set the mlflow tracking uri within the local code/workflow:
def set_mlflow_host():
    import mlflow
    return mlflow.set_tracking_uri(load_host())

## create a function that will be used to load the mlflow client:
def mlflow_client():
    from mlflow import MlflowClient

    client=MlflowClient(tracking_uri=load_host())

    return client

## create a function that will be used to load the experiment name first:
def load_exp_name():
    return exp_name

## this is the function that will load the tags of the experiment
def load_exp_tags():
    return tags


## create a function that will be used to set the experiment within mlflow workflow once created:
def set_mlflow_exp(name_exp=exp_name):
    import mlflow
    
    return mlflow.set_experiment(name=name_exp)


## create the function that will be used to load the model registered name -> for the model:
def load_model_name():
    return model_name

## create the function that will be used to load the tokenizer registered name -> for the tokenizer:
def load_model_name():
    return tokenizer_name


## function for testing model registry
def test_model_registry(name, version,client_var=mlflow_client()):
    from mlflow.exceptions import RestException
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = client_var
    try:
        client.get_model_version(name=name, version=version)
        print(f"Model '{name}' version {version} exists")
    except RestException:
        print(f" Model '{name}' version {version} not found")
        

## a function for testing model versioning or stages per model
def test_model_versioning(name, stage,client_var=mlflow_client()):

    from mlflow.exceptions import RestException
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = mlflow_client()
    try:
        client.get_model_version(name=name, stage=stage)
        print(f"Model '{name}' at {stage} exists")
    except RestException:
        print(f"Model '{name}' at {stage} not found")
