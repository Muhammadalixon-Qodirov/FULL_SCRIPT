# pipeline_modular/dataset_utils.py
import os
from collections import OrderedDict
from .data_utils import clean_text, detect_language_safe, save_jsonl

def parse_chat_file_to_instruction(raw_chat_path: str):
    out = []
    if not os.path.exists(raw_chat_path):
        return out
    with open(raw_chat_path,"r",encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i=0
    while i<len(lines)-1:
        if lines[i].lower().startswith("user:"):
            user_text = clean_text(lines[i][len("user:"):])
            if lines[i+1].lower().startswith("assistant:"):
                assistant_text = clean_text(lines[i+1][len("assistant:"):])
                if len(user_text)<3 or len(assistant_text)<1:
                    i+=2
                    continue
                lang_user = detect_language_safe(user_text)
                lang_assistant = detect_language_safe(assistant_text)
                if lang_user in ['uz','en','ru'] and lang_assistant in ['uz','en','ru']:
                    out.append({"instruction":user_text,"input":"","response":assistant_text})
                i+=2
            else:
                i+=1
        else:
            i+=1
    return out

def convert_raw_texts_to_formats(raw_text_file: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    from .data_utils import save_jsonl
    chat_rows, instr_rows, rag_rows = [], [], []
    with open(raw_text_file,"r",encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for txt in lines:
        txt_c = clean_text(txt)
        if not txt_c: continue
        chat_rows.append({"messages":[{"role":"user","content":"Iltimos, qisqacha tushuntiring:"},{"role":"assistant","content":txt_c}]})
        instr_rows.append({"instruction":"Berilgan matnni ilmiy uslubda qisqacha xulosa qiling.","input":txt_c,"response":txt_c})
        rag_rows.append({"question":"Mazmunini qisqacha tushuntiring.","context":txt_c,"answer":txt_c})
    save_jsonl(chat_rows, os.path.join(out_dir,"chat_dataset.jsonl"))
    save_jsonl(instr_rows, os.path.join(out_dir,"instruction_dataset.jsonl"))
    save_jsonl(rag_rows, os.path.join(out_dir,"rag_dataset.jsonl"))
    return chat_rows, instr_rows, rag_rows

def deduplicate_jsonl(path_in: str, path_out: str):
    seen = OrderedDict()
    with open(path_in,"r",encoding="utf-8") as fin:
        for ln in fin:
            try:
                obj = json.loads(ln)
                key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
                if key not in seen: seen[key]=obj
            except: continue
    with open(path_out,"w",encoding="utf-8") as fout:
        for obj in seen.values(): fout.write(json.dumps(obj, ensure_ascii=False)+"\n")
    print("Deduped saved:", path_out, "count:", len(seen))
