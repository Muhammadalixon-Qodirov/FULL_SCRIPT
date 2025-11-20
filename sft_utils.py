# pipeline_modular/sft_utils.py
from transformers import AutoTokenizer

PROMPT_TEMPLATE = "{instruction}\n\n{input}\n\n### Response:\n"

def build_prompt(instruction:str, input_text:str):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip(), input=input_text.strip())

def tokenize_sft_examples(examples, tokenizer, max_length):
    input_texts=[]
    for instr, inp, resp in zip(examples["instruction"], examples["input"], examples["response"]):
        prompt = build_prompt(instr, inp)
        full = prompt + resp
        input_texts.append((prompt, full))
    prompts = [p for p,f in input_texts]
    fulls = [f for p,f in input_texts]
    tok_full = tokenizer(fulls, truncation=True, padding="max_length", max_length=max_length)
    tok_prompt = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
    labels=[]
    for i in range(len(fulls)):
        input_ids = tok_full["input_ids"][i]
        prompt_ids = tok_prompt["input_ids"][i]
        prompt_len = sum(1 for id in prompt_ids if id != tokenizer.pad_token_id)
        label = [-100]*prompt_len + input_ids[prompt_len:]
        label = label[:max_length] + [-100]*max(0,max_length-len(label))
        labels.append(label)
    tok_full["labels"]=labels
    return tok_full
