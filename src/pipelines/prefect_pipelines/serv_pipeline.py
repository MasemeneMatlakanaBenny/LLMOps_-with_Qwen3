from prefect import task,flow

@task
def serving_task():
    """
    Component for serving the model and the tokenizer after they both have been productionized in mlflow
    """
    import mlflow.pyfunc
    import mlflow.transformers
    from configurations_mlflow import load_model_name,load_tokenizer_name
    from configurations_mlflow import set_mlflow_exp,set_mlflow_host

    set_mlflow_host()
    set_mlflow_exp()

    model_name=load_model_name()
    tokenizer_name=load_tokenizer_name()

    ## stage is the same:
    stage="production"

    model_uri=f"models/:{model_name}/{stage}"
    tokenizer_uri=f"models/:{tokenizer_name}/{stage}"

    model=mlflow.transformers.load_model(model_uri=model_uri)
    tokenizer=mlflow.transformers.load_model(model_uri=model_uri)

    return model,tokenizer

@flow
def serving_flow():
    model,tokenizer=serving_task()
    return model,tokenizer
