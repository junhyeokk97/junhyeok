# -*- coding: utf-8 -*-
"""
Gemma 3 QLoRA Incremental 학습 + 즉시 추론/저장
- 학습: ./data/train.jsonl / ./data/val.jsonl
- 출력: outputs/gemma3_qlora_current/ 어댑터 + outputs/preds_test.csv
"""
import os, json, re, ast, csv, inspect, warnings
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import List, Optional, Dict
import torch
from datasets import Dataset, Features, Value, Sequence
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ----------------- 설정 -----------------
MODEL_ID   = "google/gemma-3-4b-it"
TRAIN_PATH = "./data/train.jsonl"
VAL_PATH   = "./data/val.jsonl"
ADAPTER_DIR = "outputs/gemma3_qlora_current"  # incremental용

BATCH_SIZE = 2
GR_ACC     = 8
LR         = 1e-4
EPOCHS     = 1
MAX_STEPS  = 2           # 테스트용: 짧게
MAX_LEN    = 1024
CUT_INPUT  = 800
USE_4BIT   = True
SEED       = 42

MAX_NEW_TOKENS = 220

# ----------------- 프롬프트 -----------------
SYSTEM_PROMPT = (
    "당신은 전자상거래 리뷰 분석가입니다. 다음 '리뷰'를 읽고 "
    "오직 순수 JSON으로 긍정인지 부정인지 문맥을 파악하고, 감성과 긍/부정 키워드를 출력하세요."
)
JSON_SCHEMA = (
    '{ "sentiment":"긍정"|"부정", "positive_keywords":["..."], "negative_keywords":["..."] }'
)
REQUIRE_JSON_PROMPT = (
    "아래 리뷰를 읽고 반드시 이 JSON만 출력하세요. 설명/문장/코드펜스 금지.\n"
    '{ "sentiment":"긍정"|"부정", "positive_keywords":["..."], "negative_keywords":["..."] }\n'
    "- positive_keywords / negative_keywords 는 각각 최소 1개 이상 포함.\n"
)

# ----------------- 메시지 생성 -----------------
def build_messages(review: str, label: Optional[Dict]=None) -> List[Dict[str,str]]:
    user = (
        f"{SYSTEM_PROMPT}\n\n"
        f"- 키워드는 명사/구 위주, 동의어 통합\n"
        f"- 최소 한쪽 배열은 1개 이상 포함\n\n"
        f"리뷰:\n{review[:CUT_INPUT]}\n\n"
        f"출력 형식:\n{JSON_SCHEMA}"
    )
    msgs = [
        {"role":"system","content":"한국어 리뷰 분석 모델"},
        {"role":"user","content": user},
    ]
    if label is not None:
        msgs.append({"role":"assistant","content": json.dumps(label, ensure_ascii=False)})
    return msgs

# ----------------- 경고 정리 -----------------
warnings.filterwarnings("ignore", message=".*flash attention.*")
torch.set_float32_matmul_precision("high")

# ----------------- JSONL 로더 -----------------
def _sanitize_record(r: dict) -> dict:
    r["review"] = str(r.get("review", ""))
    r["doc_id"] = str(r.get("doc_id","")) if r.get("doc_id") is not None else ""
    lab = r.get("label", {}) or {}
    if not isinstance(lab, dict):
        lab = {}
    sent = lab.get("sentiment", "")
    if not isinstance(sent, str):
        sent = str(sent)
    pos = lab.get("positive_keywords", []) or []
    neg = lab.get("negative_keywords", []) or []
    pos = [str(x) for x in pos if isinstance(x, (str,int,float))]
    neg = [str(x) for x in neg if isinstance(x, (str,int,float))]
    r["label"] = {"sentiment": sent, "positive_keywords": pos, "negative_keywords": neg}
    return r

FEATS_RAW = Features({
    "review": Value("string"),
    "label": {
        "sentiment": Value("string"),
        "positive_keywords": Sequence(Value("string")),
        "negative_keywords": Sequence(Value("string")),
    },
    "doc_id": Value("string"),
})

def load_jsonl_strict(path: str) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p.resolve()}")
    records = []
    with open(p, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s: continue
            obj = None
            try:
                obj = json.loads(s)
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                except Exception: continue
            if not isinstance(obj, dict): continue
            records.append(_sanitize_record(obj))
    if not records:
        raise ValueError(f"JSONL 파싱 결과가 비었습니다: {p.resolve()}")
    ds = Dataset.from_list(records, features=FEATS_RAW)
    return ds

def to_messages_only(ds: Dataset) -> Dataset:
    def _to_msg(example):
        return {"messages": build_messages(example.get("review",""), example.get("label"))}
    return ds.map(_to_msg, remove_columns=[c for c in ds.column_names])

# ----------------- 토크나이저/모델 -----------------
print(f"[INFO] load tokenizer/model: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb_cfg = None
if USE_4BIT:
    try:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception:
        print("[WARN] bitsandbytes unavailable, fallback precision.")
        USE_4BIT = False

_dtype = "auto" if USE_4BIT else (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8) else torch.float16)

# ----------------- Incremental 학습용 모델 로드 -----------------
if Path(ADAPTER_DIR).exists():
    print("[INFO] 기존 adapter 불러오기...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=_dtype, quantization_config=bnb_cfg if USE_4BIT else None)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
else:
    print("[INFO] 새 adapter 생성...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=_dtype, quantization_config=bnb_cfg if USE_4BIT else None)
    base.config.use_cache = False
    if USE_4BIT:
        base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base, lora_cfg)

model.print_trainable_parameters()

# ----------------- 데이터셋 -----------------
train_raw = load_jsonl_strict(TRAIN_PATH)
val_raw   = load_jsonl_strict(VAL_PATH)
train_ds = to_messages_only(train_raw)
val_ds   = to_messages_only(val_raw)

# ----------------- Collator -----------------
@dataclass
class DataCollatorForChatOnlyAnswer:
    tokenizer: AutoTokenizer
    max_len: int = 1024
    def __call__(self, features):
        input_ids, attention_masks, labels = [], [], []
        for f in features:
            msgs = f["messages"]
            prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            target_text = msgs[-1]["content"]
            prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
            target_ids = tok(target_text, add_special_tokens=False).input_ids
            ids = prompt_ids + target_ids
            lab = [-100]*len(prompt_ids) + target_ids
            ids, lab = ids[:self.max_len], lab[:self.max_len]
            attn = [1]*len(ids)
            input_ids.append(torch.tensor(ids))
            attention_masks.append(torch.tensor(attn))
            labels.append(torch.tensor(lab))
        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tok.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }
        return batch

collator = DataCollatorForChatOnlyAnswer(tokenizer=tok, max_len=MAX_LEN)

# ----------------- TrainingArguments -----------------
def safe_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys()) - {"self","kwargs","**kwargs"}
    filtered = {k:v for k,v in kwargs.items() if k in allowed}
    missing = [k for k in kwargs if k not in allowed]
    if missing: print("[WARN] Unsupported TrainingArguments keys filtered:", missing)
    return TrainingArguments(**filtered)

args = safe_training_args(
    output_dir=ADAPTER_DIR,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=max(1,BATCH_SIZE//2),
    gradient_accumulation_steps=GR_ACC,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    remove_unused_columns=False,
    evaluation_strategy="steps",
    eval_steps=40,
    save_strategy="steps",
    save_steps=80,
    save_total_limit=1,
    bf16=False,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
    seed=SEED,
)

# ----------------- Trainer -----------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
)

print("[INFO] Start training (short test run)...")
trainer.train()
trainer.save_model(ADAPTER_DIR)
tok.save_pretrained(ADAPTER_DIR)
print("[DONE] saved adapter to", ADAPTER_DIR)

# ----------------- 추론 -----------------
def fallback_keywords(text: str, k: int=6):
    toks = re.findall(r"[A-Za-z가-힣0-9]{2,20}", text)
    stop = {"그리고","하지만","그냥","정말","진짜","너무","아주","매우","많이","좀","이건","저건","그건","제품","상품","배송","포장"}
    norm = [t.lower() for t in toks if t.lower() not in stop]
    cnt = Counter(norm).most_common(k)
    return [w for w,_ in cnt]

def generate_json(review: str):
    user = f"{REQUIRE_JSON_PROMPT}\n리뷰:\n{review}\n"
    msgs = [{"role":"system","content":"한국어 리뷰 분석 모델"},{"role":"user","content":user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max(260, MAX_NEW_TOKENS),
            do_sample=True, temperature=0.2, top_p=0.9,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id
        )
    txt = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"\{[\s\S]*\}", re.sub(r"```+json|```+", "", txt, flags=re.IGNORECASE))
    obj = {}
    if m:
        try: obj = json.loads(m.group(0))
        except Exception:
            try: obj = ast.literal_eval(m.group(0))
            except Exception: obj = {}
    pos, neg, sent = obj.get("positive_keywords") or [], obj.get("negative_keywords") or [], obj.get("sentiment","")
    if not pos and not neg:
        base = fallback_keywords(review, k=6)
        half = max(1,len(base)//2) if base else 1
        pos, neg = base[:half], base[half:]
    if not sent: sent = "긍정" if len(pos)>=len(neg) else "부정"
    def _norm_kw(xs):
        out = []
        for x in xs:
            if not isinstance(x,(str,int,float)): continue
            s = str(x).strip()
            if not s: continue
            s = re.sub(r"[#\"'()\[\]{}<>]", "", s)
            s = re.sub(r"\s+", " ", s)
            out.append(s[:60])
        return sorted(set(out))
    pos, neg = _norm_kw(pos), _norm_kw(neg)
    if not pos and neg: pos=[neg[0]]
    if not neg and pos: neg=[pos[0]]
    return {"sentiment":sent, "positive_keywords":pos, "negative_keywords":neg}

# 검증 샘플 20개 예측 & 저장
pred_rows=[]
for i in range(min(20, len(val_raw))):
    rev = val_raw[i]["review"]
    gold = val_raw[i].get("label",{})
    pred = generate_json(rev)
    pred_rows.append({
        "review": rev[:200],
        "gold_sent": gold.get("sentiment",""),
        "pred_sent": pred.get("sentiment",""),
        "gold_pos": ",".join(gold.get("positive_keywords",[])),
        "pred_pos": ",".join(pred.get("positive_keywords") or []),
        "gold_neg": ",".join(gold.get("negative_keywords",[])),
        "pred_neg": ",".join(pred.get("negative_keywords") or []),
    })

os.makedirs("outputs", exist_ok=True)
csv_path = "outputs/preds_test.csv"
with open(csv_path,"w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
    w.writeheader(); w.writerows(pred_rows)
print("[DONE] wrote", csv_path)
