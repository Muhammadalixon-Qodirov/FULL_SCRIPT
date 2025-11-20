# pipeline_modular/lora_utils.py
import os, torch, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from .sft_utils import tokenize_sft_examples

def train_lora_on_jsonl(jsonl_path:str, out_dir:str, config):
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"], use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    rows=[]
    with open(jsonl_path,"r",encoding="utf-8") as f:
        for ln in f: rows.append(json.loads(ln))
    ds = Dataset.from_list(rows)
    tokenized = ds.map(lambda batch: tokenize_sft_examples(batch, tokenizer, config["MAX_LENGTH"]), batched=True, remove_columns=["instruction","input","response"])
    model = AutoModelForCausalLM.from_pretrained(config["MODEL_NAME"], device_map="auto", load_in_4bit=config["LOAD_IN_4BIT"])
    model = prepare_model_for_kbit_training(model)
    lora_conf = LoraConfig(r=config["LORA_R"], lora_alpha=config["LORA_ALPHA"], target_modules=config["LORA_TARGET_MODULES"], lora_dropout=config["LORA_DROPOUT"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_conf)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=config["BATCH_SIZE"], num_train_epochs=config["EPOCHS"], learning_rate=config["LR"], fp16=torch.cuda.is_available(), logging_steps=50, save_strategy="epoch")
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, tokenizer=tokenizer, data_collator=data_collator)
    trainer.train()
    trainer.save_model(out_dir)
    print("LoRA trained:", out_dir)
    return out_dir

def merge_lora_adapters(lora_dirs, merged_outdir, normalize=False, config=None):
    assert len(lora_dirs)>0
    os.makedirs(merged_outdir, exist_ok=True)
    sd_accum={}
    total=0
    adapter_config=None
    for d in lora_dirs:
        bin_path = os.path.join(d,"pytorch_model.bin")
        cfg_path = os.path.join(d,"adapter_config.json")
        sd = torch.load(bin_path,map_location="cpu")
        if adapter_config is None and os.path.exists(cfg_path):
            adapter_config=json.load(open(cfg_path,"r",encoding="utf-8"))
        for k,v in sd.items():
            if 'lora_' in k:
                sd_accum[k]=v.clone().float() if k not in sd_accum else sd_accum[k]+v.clone().float()
        total+=1
    if normalize and total>0:
        for k in sd_accum: sd_accum[k]/=float(total)
    torch.save(sd_accum, os.path.join(merged_outdir,"pytorch_model.bin"))
    if adapter_config:
        with open(os.path.join(merged_outdir,"adapter_config.json"),"w",encoding="utf-8") as f:
            json.dump(adapter_config,f,ensure_ascii=False,indent=2)
    else:
        fallback = {"r": config["LORA_R"],"lora_alpha":config["LORA_ALPHA"],"target_modules":config["LORA_TARGET_MODULES"],"lora_dropout":config["LORA_DROPOUT"],"bias":"none","inference_mode":False}
        with open(os.path.join(merged_outdir,"adapter_config.json"),"w",encoding="utf-8") as f: json.dump(fallback,f,ensure_ascii=False,indent=2)
    print("Merged LoRA saved:", merged_outdir)
    return merged_outdir

def load_base_and_adapter(base_model_name, adapter_path=None, device="cuda", config=None):
    base = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", load_in_4bit=config["LOAD_IN_4BIT"])
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,use_fast=False)
    tokenizer.pad_token=tokenizer.eos_token
    model = PeftModel.from_pretrained(base, adapter_path, device_map="auto") if adapter_path else base
    print("Loaded model:", adapter_path if adapter_path else base_model_name)
    model.eval()
    return model, tokenizer

def generate_with_model(model, tokenizer, prompt, max_new_tokens=200, device="cuda", config=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config["MAX_LENGTH"]).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)
