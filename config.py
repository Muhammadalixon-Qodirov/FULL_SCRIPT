# pipeline_modular/config.py
import os
import torch

CONFIG = {
    "RAW_CHAT_FILE": "user_chats.txt",
    "RAW_TEXT_FILE": "raw_texts.txt",
    "OUTPUT_DIR": "output",
    "CLEANED_JSONL": "output/cleaned.jsonl",
    "INSTR_JSONL": "output/instruction_dataset.jsonl",
    "CHAT_JSONL": "output/chat_dataset.jsonl",
    "RAG_JSONL": "output/rag_dataset.jsonl",
    "MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
    "LOAD_IN_4BIT": True,
    "MAX_LENGTH": 2048,
    "BATCH_SIZE": 1,
    "EPOCHS": 3,
    "LR": 2e-4,
    "LORA_R": 16,
    "LORA_ALPHA": 32,
    "LORA_DROPOUT": 0.05,
    "LORA_TARGET_MODULES": ["q_proj","v_proj","k_proj","o_proj"],
    "LORA_OUT_DIR": "lora_adapters",
    "MERGED_LORA_DIR": "lora_merged",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
os.makedirs(CONFIG["LORA_OUT_DIR"], exist_ok=True)
