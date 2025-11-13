## creating the pipeline for evaluatinf the generated text:

from kfp import dsl,compiler
from kfp.dsl import pipeline,component

@component
def evaluation_component(input_prompt):
    import evaluate
    from chat_with_model import response_model
    from serv_pipeline import serving_pipeline

    ## get the model and the tokenizer first:
    model,tokenizer=serving_pipeline()

    ## get the generated text:
    response=response_model(model=model,tokenizer=tokenizer,prompt=input_prompt)

    perplexity=evaluate.load("perplexity")


    checkpoint="qwen/qwen3"

    metrics=perplexity.compute(
        model_id=checkpoint,
        add_start_token=True,
        references=[input_prompt],
        predictions=[response]
        )
    
    return metrics


@pipeline(
    name="evalaution_pipeline",
    description="Relying on Perplexity for model evaluation"
)
def evaluation_pipeline(input_prompt):
    
    model_metrics=evaluation_component(input_prompt=input_prompt)

    return model_metrics

compiler.Compiler().compile(
    pipeline_func=evaluation_pipeline,
    package_path="pipelines/kubeflow_pipelines/evaluation_pipeline_file.yaml"
)

