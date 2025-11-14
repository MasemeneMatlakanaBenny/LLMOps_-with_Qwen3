from flask import Flask
from chat_with_model import response_model
from pipelines.kubeflow_pipelines.serv_pipeline import serving_pipeline


## get the prompt:
prompt=input("Enter the prompt here: ")
## create the app :
app=Flask(__name__)

## get the model and tokenizer:
model,tokenizer=serving_pipeline()



## now:
@app.route('/response')
def model_generated_text(model=model,tokenizer=tokenizer,methods=['POST']):

    generated_text=response_model(model=model,tokenizer=tokenizer,prompt=prompt)

    return generated_text


@app.route('/metrics')
def metrics(generated_text,prompt=prompt):
    import evaluate
    
    perplexity=evaluate.load("perplexity")


    checkpoint="qwen/qwen3"

    metrics=perplexity.compute(
        model_id=checkpoint,
        add_start_token=True,
        references=[prompt],
        predictions=[generated_text]
        )
    
    return metrics




