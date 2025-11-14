
from prefect import task,flow

@task
def evaluation_task(input_prompt):
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


@task
def evaluation_flow():
    metrics=evaluation_task()

    return metrics
