# pipeline_modular/data_utils.py
import re, json
from langdetect import detect

PII_PATTERNS = [
    (re.compile(r'\b\d{12,19}\b'), '[CARD]'),
    (re.compile(r'\+?\d{7,15}'), '[PHONE]'),
    (re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', flags=re.I), '[EMAIL]'),
    (re.compile(r'\b[A-Z]{2}\d{6,9}\b'), '[ID]'),
]

def remove_pii(text: str) -> str:
    if not text: return text
    t = text
    for pat, repl in PII_PATTERNS:
        t = pat.sub(repl, t)
    return t

def clean_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text: str) -> str:
    if text is None: return ""
    t = text.strip()
    t = remove_pii(t)
    t = clean_whitespace(t)
    return t

def detect_language_safe(text: str):
    try:
        return detect(text)
    except:
        return None

def save_jsonl(data:list, path:str):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("Saved", path, "rows:", len(data))
