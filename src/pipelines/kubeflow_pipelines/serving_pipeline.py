## creating a kubeflow pipeline that allows us to serve the model:
# import the libs first:
from kfp import dsl,compiler
from kfp.dsl import pipeline,component

@component
def serving_component():
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

## create the pipeline for serving the model:
@pipeline(
    name="serving_qwen3_pipeline",
    description="pipeline for serving the model and the tokenizer"
)
def serving_pipeline():
    """"
    The serving pipeline for the model
    """

    model,tokenizer=serving_component()

    return model,tokenizer

compiler.Compiler().compile(
    pipeline_func=serving_pipeline,
    package_path="pipelines/kubeflow_pipelines/serving_pipeline_file.yaml"
)



