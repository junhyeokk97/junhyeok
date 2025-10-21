# -*- coding: utf-8 -*-
"""
경량 실행용: 문맥 기반 리뷰 감성 → 긍/부정 키워드 집계 (Gemma 3 로컬, Transformers, tqdm)
- OpenAI API 완전 제거 (비용 0원)
- 대용량일 때 빠르게 '샘플만' 돌려보는 옵션 포함
"""

import os, re, json, time, uuid, datetime, random, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from collections import Counter
from tqdm import tqdm
from transformers import logging as hf_logging
import torch
torch.set_float32_matmul_precision("high")

hf_logging.set_verbosity_error()  # 선택: HF 경고 숨기기
import ast
import torch
torch.set_float32_matmul_precision("high")
# ----------------- 여기를 조정하세요 -----------------
CONFIG = {
    # 무료 최신 모델(한글 가능). 더 가볍게는 phi-3-mini 등도 가능.
    "HF_MODEL_ID": "google/gemma-3-4b-it",

    # 프로젝트 루트 (아래 이름의 폴더들을 자동 탐색)
    "PROJECT_ROOT": r"C:\study25\project\Sample",

    # ↓↓↓ 속도/자원 줄이는 핵심 옵션 ↓↓↓
    "MAX_FILES": 20,               # 최신 파일 상위 N개만 사용 (None이면 모든 파일)
    "PER_FILE_MAX_ROWS": 1000,     # 파일당 최대 행 수
    "GLOBAL_MAX_ROWS": 10000,      # 전체 최대 행 수
    "SAMPLE_STRATEGY": "random",   # "head" | "random" | "stratified:카테고리"
    "RANDOM_SEED": 42,

    "MIN_TEXT_LEN": 15,            # 이 길이 미만 리뷰 제거
    "MAX_TEXT_CHARS": 800,         # LLM 입력 텍스트 길이 제한
    "MAX_NEW_TOKENS": 220,         # LLM의 최대 출력 토큰
    "MAX_CALLS": 10000,            # 분석 상한(테스트용). 로컬이라 비용은 없음.

    # (선택) 강제 컬럼 명시
    "TEXT_COL_OVERRIDE": "상품평",     # 예: "상품평"
    "ID_COL_OVERRIDE": "INDEX",       # 예: "INDEX"

    # (선택) 메모리 절약: 4bit 양자화 사용 여부
    "USE_4BIT": True,             # True로 두면 GPU VRAM/메모리 절약 (bitsandbytes 필요)
}
# -----------------------------------------------------

OUT_DIR = Path("./outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
EXT_ALLOW = {".xlsx", ".json"}

# ---------- Transformers (Gemma 3) ----------
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

HF_MODEL_ID = CONFIG["HF_MODEL_ID"]
from transformers import BitsAndBytesConfig
qcfg = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, quantization_config=qcfg, device_map="auto")
# 4bit 설정(선택)
bnb_cfg = None
if CONFIG["USE_4BIT"]:
    try:
        from bitsandbytes import __version__ as _bbv  # noqa
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
    except Exception:
        print("[WARN] bitsandbytes가 없어 4bit를 비활성화합니다.")
        CONFIG["USE_4BIT"] = False

print(f"[INFO] Load model: {HF_MODEL_ID} (4bit={CONFIG['USE_4BIT']})")
tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=bnb_cfg if CONFIG["USE_4BIT"] else None,
)

# ---------- 탐색 ----------
def find_target_dirs(root: Path) -> List[Path]:
    targets, seen = [], set()
    for p in root.rglob("*"):
        if p.is_dir():
            n = p.name.strip()
            if n.startswith("01.원천데이터") or n.startswith("02.라벨링데이터"):
                sp = str(p.resolve())
                if sp not in seen:
                    targets.append(p); seen.add(sp)
    return targets

def find_files_auto(project_root: str) -> List[Path]:
    root = Path(project_root)
    if not root.exists():
        raise FileNotFoundError(f"프로젝트 루트를 찾을 수 없습니다: {project_root}")
    tdirs = find_target_dirs(root)
    if not tdirs:
        print("[WARN] 타깃 폴더를 못 찾아 루트 전체를 스캔합니다.")
        tdirs = [root]
    files = []
    for td in tdirs:
        for f in td.rglob("*"):
            if f.is_file() and f.suffix.lower() in EXT_ALLOW:
                files.append(f)
    # 최신 수정일 기준 정렬(내림차순) 후 상위 N개 제한
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if CONFIG["MAX_FILES"]:
        files = files[: int(CONFIG["MAX_FILES"])]
    return files

# ---------- 로더 & 스키마 ----------
def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            key = next((k for k,v in data.items() if isinstance(v, list)), None)
            df = pd.json_normalize(data[key]) if key else pd.json_normalize(data)
        else:
            df = pd.DataFrame([data])
    df["__source_file"] = str(path)
    # 파일당 행 수 제한
    if CONFIG["PER_FILE_MAX_ROWS"]:
        df = df.head(int(CONFIG["PER_FILE_MAX_ROWS"]))
    return df

# 텍스트/ID 후보 컬럼(‘상품평’ 포함)
CAND_TEXT = [
    "상품평","리뷰","리뷰내용","내용","본문","후기",
    "text","content","review","comment","body","message","caption"
]
CAND_ID   = ["id","review_id","리뷰아이디","리뷰ID","uid","_id","doc_id","INDEX","index"]

def pick_col(cols: List[str], cands: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    norm = {re.sub(r"\s+","",c.lower()): c for c in cols}
    for c in cands:
        k = re.sub(r"\s+","",c.lower())
        for nk, orig in norm.items():
            if k in nk:
                return orig
    return None

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    tcol = CONFIG["TEXT_COL_OVERRIDE"] if (CONFIG["TEXT_COL_OVERRIDE"] in cols) else None
    icol = CONFIG["ID_COL_OVERRIDE"]   if (CONFIG["ID_COL_OVERRIDE"]   in cols) else None
    if tcol is None:
        tcol = pick_col(cols, CAND_TEXT)
    if icol is None:
        icol = pick_col(cols, CAND_ID)
    if tcol is None:
        raise ValueError(f"리뷰 텍스트 컬럼을 찾지 못했습니다. 후보={CAND_TEXT} / 실제={cols[:15]}")
    out = pd.DataFrame({
        "doc_id": df[icol] if (icol is not None and icol in df) else [str(uuid.uuid4()) for _ in range(len(df))],
        "text": df[tcol].astype(str).fillna(""),
    })
    # 최소 글자수 필터
    out = out[out["text"].str.len() >= int(CONFIG["MIN_TEXT_LEN"])]
    return out

# ---------- 샘플링 유틸 ----------
def stratified_sample(df: pd.DataFrame, by: str, n: int, seed: int = 42) -> pd.DataFrame:
    if by not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=seed)
    groups = list(df.groupby(by))
    per_group = max(1, n // max(1, len(groups)))
    parts = []
    for _, g in groups:
        take = min(len(g), per_group)
        parts.append(g.sample(n=take, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    # 부족하면 랜덤으로 채우기
    if len(out) < n:
        remain = df.drop(out.index, errors="ignore")
        if len(remain) > 0:
            extra = remain.sample(n=min(n-len(out), len(remain)), random_state=seed)
            out = pd.concat([out, extra], ignore_index=True)
    return out

def sample_rows(df: pd.DataFrame, target_n: Optional[int]) -> pd.DataFrame:
    if not target_n or len(df) <= target_n:
        return df
    random.seed(CONFIG["RANDOM_SEED"])
    strat = CONFIG["SAMPLE_STRATEGY"]
    if strat.startswith("stratified:"):
        col = strat.split(":",1)[1].strip()
        return stratified_sample(df, by=col, n=target_n, seed=CONFIG["RANDOM_SEED"])
    elif strat == "head":
        return df.head(target_n)
    else:  # random
        return df.sample(n=target_n, random_state=CONFIG["RANDOM_SEED"])

# ---------- 프롬프트 & LLM 호출 ----------
PROMPT_SYS = (
    "당신은 전자상거래 리뷰 분석가입니다. 입력된 '한 개의 리뷰'를 읽고:\n"
    "1) 전반적 감성을 '긍정' 또는 '부정' 중 하나로 판단\n"
    "2) 해당 리뷰 문맥에서 '긍정 키워드'와 '부정 키워드'를 분리해 1~6개씩 추출\n"
    "- 키워드는 명사/구(브랜드, 품질, 핏, 소재, 색상, 배송/포장, 가격/가성비, 재구매의사 등)\n"
    "- 중복/동의어는 하나로 통합(예: 가성비/가격은 '가격'으로)\n"
    "- 두 배열이 동시에 비어 있어서는 안 됨. 최소 한쪽은 1개 이상 포함\n"
    "- 오직 '순수 JSON'만 출력. 설명/문장/코드펜스(```` 등) 금지\n"
    '{\n'
    '  "sentiment": "긍정" | "부정",\n'
    '  "positive_keywords": ["키워드1", "..."],\n'
    '  "negative_keywords": ["키워드1", "..."]\n'
    "}"
)

def extract_json_block(text: str):
    """모델 출력에서 순수 JSON 객체만 뽑아낸다(코드펜스/여분 텍스트 허용)."""
    # 코드펜스 제거
    text = re.sub(r"```+json|```+", "", text, flags=re.IGNORECASE)
    # 모든 JSON 후보 블록 수집
    cands = re.findall(r"\{[\s\S]*?\}", text)
    # 뒤에서부터 시도(정답일 확률↑)
    for block in reversed(cands):
        try:
            return json.loads(block)
        except Exception:
            # json.loads 실패 시, ast.literal_eval로 폴백
            try:
                return ast.literal_eval(block)
            except Exception:
                continue
    return None

def run_gemma(messages: List[Dict[str, str]], max_new_tokens: int = 220) -> str:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,          # 소량 샘플링 허용 (빈 응답 방지)
        temperature=0.2,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out_ids[0], skip_special_tokens=True)

# 간단 불용어/극성 어휘 (필요시 확장 가능)
STOPWORDS = set(["그리고","하지만","그냥","정말","진짜","너무","아주","매우","많이","좀",
                 "이건","저건","그건","제품","상품","배송","포장"])
POS_HINTS = set(["좋","예쁘","만족","추천","빠르","맘에","가볍","편하","괜찮"])
NEG_HINTS = set(["불량","느리","늦","아쉽","별로","최악","불편","작","큼","무겁","비싸","싸구려","환불","교환"])

def fallback_keywords(text: str, k: int = 4):
    """모델이 비우는 경우 대비: 간단 토큰 빈도 기반 키워드 상위 k개."""
    toks = re.findall(r"[A-Za-z가-힣0-9]{2,20}", text)
    norm = []
    for t in toks:
        nt = t.lower()
        if nt in STOPWORDS:
            continue
        norm.append(nt)
    freq = Counter(norm).most_common()
    return [w for w, _ in freq[:k]]

def analyze_review(text: str) -> Dict[str, Any]:
    messages = [
        {"role":"system","content": PROMPT_SYS},
        {"role":"user","content": text[: int(CONFIG['MAX_TEXT_CHARS'])]}
    ]
    raw = run_gemma(messages, max_new_tokens=int(CONFIG["MAX_NEW_TOKENS"]))

    # 1) 견고한 JSON 추출
    obj = extract_json_block(raw) or {}

    # 2) 키워드/감성 꺼냄
    pos = obj.get("positive_keywords") or []
    neg = obj.get("negative_keywords") or []
    sent = obj.get("sentiment", "")

    # 3) 키워드 정규화
    def norm_kw(x: str) -> str:
        x = x.strip()
        x = re.sub(r"[#\"'()\[\]{}<>]", "", x)
        x = re.sub(r"\s+", " ", x)
        return x[:60]
    pos = sorted({norm_kw(k) for k in pos if isinstance(k, str) and k.strip()})
    neg = sorted({norm_kw(k) for k in neg if isinstance(k, str) and k.strip()})

    # 4) 둘 다 비면 → 규칙 기반 백업
    if not pos and not neg:
        base = fallback_keywords(text, k=6)
        p, n = [], []
        # 극성 힌트에 따라 분배
        pos_hits = sum(h in text for h in POS_HINTS)
        neg_hits = sum(h in text for h in NEG_HINTS)
        if pos_hits or neg_hits:
            for w in base:
                if pos_hits: p.append(w)
                if neg_hits: n.append(w)
        else:
            # 힌트가 없으면 절반 분배
            half = max(1, len(base)//2) if base else 1
            p, n = base[:half], base[half:]
        pos, neg = sorted(set(p)), sorted(set(n))

    # 5) 감성 라벨 보강
    if sent not in ("긍정","부정"):
        pos_hits = sum(h in text for h in POS_HINTS)
        neg_hits = sum(h in text for h in NEG_HINTS)
        if pos_hits > neg_hits: sent = "긍정"
        elif neg_hits > pos_hits: sent = "부정"
        else: sent = "긍정" if len(pos) >= len(neg) else "부정"

    # 6) 최소 1개는 보장 (한쪽이 비면 반대에서 하나 복사)
    if not pos and neg: pos = [neg[0]]
    if not neg and pos: neg = [pos[0]]

    # (선택) 디버그 로그
    # dump_debug("<doc_id>", text, raw, {"sentiment": sent, "pos": pos, "neg": neg})

    return {"sentiment": sent, "positive_keywords": pos, "negative_keywords": neg}

# ---------- 집계 & 저장 ----------
def main():
    random.seed(CONFIG["RANDOM_SEED"])
    print(f"[INFO] HF model = {CONFIG['HF_MODEL_ID']}")
    files = find_files_auto(CONFIG["PROJECT_ROOT"])
    print(f"[INFO] 선택된 파일 수(최신 우선) = {len(files)}")
    if not files:
        raise SystemExit("입력 파일 없음")

    frames = []
    for p in tqdm(files, desc="Loading files", unit="file"):
        try:
            df = read_any(p)
            df2 = normalize_schema(df)
            frames.append(df2)
        except Exception as e:
            print(f"[WARN] {p}: {e}")

    if not frames:
        raise SystemExit("적재 가능한 리뷰가 없습니다.")

    data = pd.concat(frames, ignore_index=True)
    data = data[data["text"].astype(str).str.strip().ne("")]

    # (선택) 글로벌 상한으로 샘플링
    if CONFIG["GLOBAL_MAX_ROWS"]:
        data = sample_rows(data, int(CONFIG["GLOBAL_MAX_ROWS"]))

    # 중복 제거 (문장 해시 기반)
    def hash_text(x: str) -> str:
        return hashlib.md5(x.encode("utf-8")).hexdigest()
    data["__h"] = data["text"].apply(hash_text)
    data = data.drop_duplicates(subset=["__h"]).drop(columns=["__h"]).reset_index(drop=True)

    print(f"[INFO] 최종 분석 대상 리뷰 수: {len(data)}")

    rows, calls = [], 0
    t0 = time.time()
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Analyzing reviews", unit="review"):
        if CONFIG["MAX_CALLS"] and calls >= int(CONFIG["MAX_CALLS"]):
            break
        res = analyze_review(str(row["text"]))
        rows.append({
            "doc_id": row["doc_id"],
            "sentiment": res["sentiment"],
            "pos_keywords": json.dumps(res["positive_keywords"], ensure_ascii=False),
            "neg_keywords": json.dumps(res["negative_keywords"], ensure_ascii=False),
        })
        calls += 1

    per_df = pd.DataFrame(rows)

    # 키워드 폭발 → 집계
    pos_cnt, neg_cnt = Counter(), Counter()
    for _, r in per_df.iterrows():
        try:
            for k in json.loads(r["pos_keywords"]):
                pos_cnt[k] += 1
        except: pass
        try:
            for k in json.loads(r["neg_keywords"]):
                neg_cnt[k] += 1
        except: pass

    pos_df = pd.DataFrame([{"keyword": k, "count": c} for k, c in pos_cnt.most_common()])
    neg_df = pd.DataFrame([{"keyword": k, "count": c} for k, c in neg_cnt.most_common()])

    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    per_path = OUT_DIR / f"per_review_{dt}.csv"
    pos_path = OUT_DIR / f"keywords_positive_{dt}.csv"
    neg_path = OUT_DIR / f"keywords_negative_{dt}.csv"

    per_df.to_csv(per_path, index=False, encoding="utf-8-sig")
    pos_df.to_csv(pos_path, index=False, encoding="utf-8-sig")
    neg_df.to_csv(neg_path, index=False, encoding="utf-8-sig")

    print("[DONE] 저장 파일:")
    print(f"- {per_path}")
    print(f"- {pos_path}")
    print(f"- {neg_path}")
    print(f"[TIME] {(time.time()-t0):.1f}s / 처리 {calls}건 / 출력 {len(per_df)}행")

if __name__ == "__main__":
    main()
