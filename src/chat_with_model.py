## create a function that allows the user to receive a response from the model:
import transformers
def response_model(model:transformers.AutoModelForCausalLM,tokenizer:transformers.AutoTokenizer,prompt):
    """
    Getting the response of the model.
    Enter some prompt to the model ->   
    Get the input id ->
    Use input id to get -> attention mask + input ids
    Use the attention mask + input ids -> get the generated output
    -> decode the generated output using the tokenizer
    """
    inputs=tokenizer(prompt,return_tensors="pt")

    input_ids=inputs.input_ids
    attention_mask=inputs.attention_mask

    ## get the output:
    output=model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_num_sequences=1,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.7
    )

    generated_text=tokenizer.decode(output,skip_special_tokens=True)

    return generated_text


