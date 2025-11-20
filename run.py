from pipeline_modular import config, data_utils, dataset_utils, lora_utils, sft_utils
# 1) Chat faylni parse qilish
chats = dataset_utils.parse_chat_file_to_instruction(config.CONFIG["RAW_CHAT_FILE"])
data_utils.save_jsonl(chats, config.CONFIG["CLEANED_JSONL"])

# 2) LoRA train qilish
lora_dir = lora_utils.train_lora_on_jsonl(config.CONFIG["INSTR_JSONL"], "output/lora1", config.CONFIG)

# 3) LoRA merge
merged = lora_utils.merge_lora_adapters(["output/lora1","output/lora2"], config.CONFIG["MERGED_LORA_DIR"], normalize=True, config=config.CONFIG)

# 4) Inference
model, tokenizer = lora_utils.load_base_and_adapter(config.CONFIG["MODEL_NAME"], merged, config=config.CONFIG)
res = lora_utils.generate_with_model(model, tokenizer, "Berilgan matnni xulosa qil:")
print(res)
