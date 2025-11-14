import torch
import transformers


checkpoint="qwen/qwen3"
## lets create a device function that will be used to set the device to either gpu or cpu depending on the availability

def set_device()->torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    else:
        return torch.device("cpu")


## create a function that will be used to load the model:

def load_model(model_name:str=checkpoint,device:torch.device=set_device())->transformers.AutoModelForCausalLM:
    from transformers import AutoModelForCausalLM

    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto"
    )

    model.config.pad_token_id=model.config.eos_token_id

    return model.to(device)

## create a function that will be used to load the model in a similar manner:
def load_tokenizer(model_name:str=checkpoint,device:torch.device=set_device())->transformers.AutoTokenizer:
    from transformers import AutoTokenizer

    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

    ## set the tokenizer's token id:
    tokenizer.pad_token=tokenizer.eos_token

    return tokenizer

