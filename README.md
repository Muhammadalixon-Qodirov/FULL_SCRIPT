

## **Umumiy konsept**

Sizning eski `pipeline_full.py` kodi **bitta katta faylda barcha vazifalarni bajarayotgan**: text cleaning, dataset tayyorlash, LoRA training, LoRA merge va inference.

Men uni **modularga ajratdim**, shunda har bir vazifa alohida modulga tushadi va siz istagan funksiyani alohida chaqirishingiz mumkin.

---

## **1️⃣ config.py**

```python
CONFIG = {
    "RAW_CHAT_FILE": "user_chats.txt",
    "RAW_TEXT_FILE": "raw_texts.txt",
    ...
}
```

**Vazifasi:**

* Barcha sozlamalarni (paths, model nomi, batch size, LoRA parametrlar va hokazo) **bir joyda saqlaydi**.
* Masalan, agar siz model nomini o‘zgartirmoqchi bo‘lsangiz, faqat config.py ni o‘zgartirasiz.

**Qanday ishlatish:**

```python
from pipeline_modular import config
print(config.CONFIG["MODEL_NAME"])
```

---

## **2️⃣ data_utils.py**

```python
# text cleaning, PII removal, language detection
```

**Vazifasi:**

1. **Text cleaning** – keraksiz bo‘sh joylar, tablar, line-breaklarni olib tashlaydi.
2. **PII removal** – kredit karta, telefon raqami, email, passport kabi shaxsiy ma’lumotlarni `[CARD]`, `[PHONE]` tarzida mask qiladi.
3. **Language detection** – matn qaysi til (uz, en, ru) ekanligini aniqlaydi.
4. **save_jsonl** – datasetlarni `.jsonl` formatida saqlaydi.

**Qanday ishlatish:**

```python
from pipeline_modular import data_utils
cleaned = data_utils.clean_text("Salom, mening emailim test@mail.com")
print(cleaned)  # Salom, mening emailim [EMAIL]
```

---

## **3️⃣ dataset_utils.py**

```python
# chat/instruction/RAG dataset tayyorlash
```

**Vazifasi:**

1. **parse_chat_file_to_instruction** – `user_chats.txt` fayldan `{"instruction":..., "input":..., "response":...}` ko‘rinishida instruction dataset yaratadi.
2. **convert_raw_texts_to_formats** – `raw_texts.txt` fayldan:

   * Chat dataset (`messages`)
   * Instruction dataset (`instruction, input, response`)
   * RAG dataset (`question, context, answer`)
3. **deduplicate_jsonl** – bir xil yozuvlarni olib tashlaydi, faqat noyob yozuvlarni saqlaydi.

**Qanday ishlatish:**

```python
from pipeline_modular import dataset_utils, data_utils, config

chats = dataset_utils.parse_chat_file_to_instruction(config.CONFIG["RAW_CHAT_FILE"])
data_utils.save_jsonl(chats, config.CONFIG["CLEANED_JSONL"])
```

---

## **4️⃣ sft_utils.py**

```python
# SFT prompt + tokenization
```

**Vazifasi:**

1. Instruction datasetni **SFT formatiga** o‘tkazadi.
2. `build_prompt` – `<instruction>\n\n<input>\n\n### Response:\n` formatida prompt yaratadi.
3. `tokenize_sft_examples` – HuggingFace tokenizer yordamida matnlarni **tokenlarga ajratadi** va label yaratadi (`-100` bilan prompt qismi mask qilinadi).

**Qanday ishlatish:**

```python
from pipeline_modular import sft_utils
prompt = sft_utils.build_prompt("Qisqacha tushuntiring", "Matn contenti")
print(prompt)
```

---

## **5️⃣ lora_utils.py**

```python
# LoRA training, merge, inference
```

**Vazifasi:**

1. **train_lora_on_jsonl** – instruction datasetni LoRA adapter yordamida **fine-tuning qiladi**.
2. **merge_lora_adapters** – bir nechta LoRA adapterlarni **bitta adapterga qo‘shadi** (sum yoki average).
3. **load_base_and_adapter** – bazaviy model va adapterni yuklaydi.
4. **generate_with_model** – yuklangan model bilan **text generation** qiladi.

**Qanday ishlatish:**

```python
from pipeline_modular import lora_utils, config

# LoRA train
lora_dir = lora_utils.train_lora_on_jsonl(config.CONFIG["INSTR_JSONL"], "output/lora1", config.CONFIG)

# LoRA merge
merged_dir = lora_utils.merge_lora_adapters(["output/lora1","output/lora2"], config.CONFIG["MERGED_LORA_DIR"], normalize=True, config=config.CONFIG)

# Inference
model, tokenizer = lora_utils.load_base_and_adapter(config.CONFIG["MODEL_NAME"], merged_dir, config=config.CONFIG)
res = lora_utils.generate_with_model(model, tokenizer, "Berilgan matnni xulosa qil:")
print(res)
```

---

### **Natija**

Endi siz:

* **data_utils.py** – faqat text cleaning va PII removal
* **dataset_utils.py** – dataset yaratish / deduplicate
* **sft_utils.py** – prompt va tokenization
* **lora_utils.py** – LoRA train, merge, inference

har birini alohida ishlatishingiz mumkin.

---

