# -*- coding: utf-8 -*-
"""
01·02 폴더에서 리뷰를 모아 약식 라벨(감성/키워드) 생성 → train/val JSONL
테스트용 경량 파이프라인 (규칙 기반 라벨링)
"""
import os, re, json, uuid, random, hashlib
from pathlib import Path
from typing import List, Optional
import pandas as pd

# ------------ 설정 ------------
PROJECT_ROOT = r"C:\study25\project\Sample"
OUT_DIR = Path("data"); OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_N, VAL_N = 200, 50           # 샘플 수(테스트용 소량)
TEXT_COL_OVERRIDE = "상품평"        # 없으면 자동탐지
MIN_TEXT_LEN = 15
EXT_ALLOW = {".xlsx", ".json"}
RANDOM_SEED = 42

# 후보 컬럼들
CAND_TEXT = ["상품평","리뷰","리뷰내용","내용","본문","후기",
             "text","content","review","comment","body","message","caption"]
CAND_ID   = ["id","review_id","리뷰아이디","리뷰ID","uid","_id","doc_id","INDEX","index"]

# 간단 불용어/극성 어휘
STOPWORDS = set(["그리고","하지만","그냥","정말","진짜","너무","아주","매우","많이","좀",
                 "이건","저건","그건","제품","상품","배송","포장"])
POS_HINTS = set(["좋","예쁘","만족","추천","빠르","맘에","가볍","편하","괜찮","재구매"])
NEG_HINTS = set(["불량","느리","늦","아쉽","별로","최악","불편","작","큼","무겁","비싸","싸구려","환불","교환","파손","오염"])

# ------------ 유틸 ------------
def find_target_dirs(root: Path) -> List[Path]:
    t = []
    for p in root.rglob("*"):
        if p.is_dir() and (p.name.strip().startswith("01.원천데이터") or p.name.strip().startswith("02.라벨링데이터")):
            t.append(p)
    return t or [root]

def pick_col(cols: List[str], cands: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    norm = {re.sub(r"\s+","",c.lower()): c for c in cols}
    for c in cands:
        k = re.sub(r"\s+","",c.lower())
        for nk, orig in norm.items():
            if k in nk: return orig
    return None

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".xlsx":
        df = pd.read_excel(path)
    else:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            key = next((k for k,v in data.items() if isinstance(v, list)), None)
            df = pd.json_normalize(data[key]) if key else pd.json_normalize(data)
        else:
            df = pd.DataFrame([data])
    df["__source_file"] = str(path)
    return df

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    tcol = TEXT_COL_OVERRIDE if TEXT_COL_OVERRIDE in cols else pick_col(cols, CAND_TEXT)
    icol = pick_col(cols, CAND_ID)
    if not tcol:
        raise ValueError(f"리뷰 텍스트 컬럼을 찾지 못함. 실제={cols[:15]}")
    out = pd.DataFrame({
        "doc_id": (df[icol].astype(str) if icol in df else [str(uuid.uuid4()) for _ in range(len(df))]),
        "text": df[tcol].astype(str).fillna("")
    })
    out = out[out["text"].str.len()>=MIN_TEXT_LEN]
    return out

def fallback_keywords(text: str, k: int = 4):
    toks = re.findall(r"[A-Za-z가-힣0-9]{2,20}", text)
    norm = []
    for t in toks:
        nt = t.lower()
        if nt in STOPWORDS: 
            continue
        norm.append(nt)
    freq = {}
    for w in norm: freq[w]=freq.get(w,0)+1
    return [w for w,_ in sorted(freq.items(), key=lambda x:-x[1])[:k]]

def weak_label(text: str):
    pos_hits = sum(h in text for h in POS_HINTS)
    neg_hits = sum(h in text for h in NEG_HINTS)
    base = fallback_keywords(text, k=6)
    # 간단 분배
    if pos_hits or neg_hits:
        p = base[: max(1, len(base)//2)] if pos_hits else []
        n = base[max(1, len(base)//2):] if neg_hits else []
    else:
        half = max(1, len(base)//2) if base else 1
        p, n = base[:half], base[half:]
    sent = "긍정" if pos_hits>neg_hits or (len(p)>=len(n)) else "부정"
    return {"sentiment": sent, "positive_keywords": sorted(set(p)), "negative_keywords": sorted(set(n))}

# ------------ 메인 ------------
def main():
    random.seed(RANDOM_SEED)
    root = Path(PROJECT_ROOT)
    tdirs = find_target_dirs(root)

    files = []
    for td in tdirs:
        for f in td.rglob("*"):
            if f.is_file() and f.suffix.lower() in EXT_ALLOW:
                files.append(f)
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    frames = []
    for p in files:
        try:
            df = read_any(p)
            frames.append(normalize_schema(df))
        except Exception as e:
            print(f"[WARN] {p}: {e}")

    if not frames:
        raise SystemExit("적재 가능한 리뷰가 없습니다.")

    data = pd.concat(frames, ignore_index=True)
    # 중복 제거
    data["__h"] = data["text"].apply(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    data = data.drop_duplicates(subset="__h").drop(columns="__h").reset_index(drop=True)

    # 샘플링
    total_n = min(len(data), TRAIN_N+VAL_N)
    data = data.sample(n=total_n, random_state=RANDOM_SEED).reset_index(drop=True)

    # 약식 라벨 생성
    recs = []
    for _, r in data.iterrows():
        lab = weak_label(r["text"])
        recs.append({"review": str(r["text"]),
                    "label": lab,
                    "doc_id": str(r["doc_id"])})

    # split
    train = recs[: min(TRAIN_N, len(recs))]
    val   = recs[min(TRAIN_N, len(recs)) : min(TRAIN_N+VAL_N, len(recs))]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR/"train.jsonl","w",encoding="utf-8") as f:
        for x in train: f.write(json.dumps(x, ensure_ascii=False)+"\n")
    with open(OUT_DIR/"val.jsonl","w",encoding="utf-8") as f:
        for x in val: f.write(json.dumps(x, ensure_ascii=False)+"\n")

    print(f"[DONE] data/train.jsonl={len(train)}  data/val.jsonl={len(val)}")

if __name__ == "__main__":
    main()
