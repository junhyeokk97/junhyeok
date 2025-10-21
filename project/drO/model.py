#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review FULL One-File (FIXED)
- AI Hub 속성기반 감정분석 → 학습/평가 → 리뷰 DB 주입 → 세분 자동답글(의도 태그/심각도) → 콘솔 리포트/샘플 추론
- 경고/호환 이슈 해결: read_json encoding 인자 제거, Excel 지원, 단일 파일 로딩(--file), 컬럼 수동 매핑 옵션 추가

빠른 설치(한 줄)
  pip install pandas scikit-learn tqdm python-dateutil joblib openpyxl xlrd==1.2.0

권장 파이썬: 3.10 ~ 3.12 (최소 3.9)

예시(Windows)
  # 단일 엑셀 파일 로딩(시트명 'Sheet1' 가정) → 미리보기
  python review_full_onefile_fixed.py --file "C:\\study25\\project\\1.패션.xlsx" --sheet Sheet1 --preview

  # 디렉토리 전체(재귀) 로딩 + 학습/평가 + 모델 저장
  python review_full_onefile_fixed.py --data "C:\\study25\\project" --train --eval --out .\\_save\\attrsent

  # DB 초기화 → 임포트 → 큐 생성 → 미리보기 → 전일 리포트
  python review_full_onefile_fixed.py --init-db --db .\\review_bot.sqlite
  python review_full_onefile_fixed.py --file "C:\\study25\\project\\1.패션.xlsx" --sheet Sheet1 --import-to-db --db .\\review_bot.sqlite
  python review_full_onefile_fixed.py --file "C:\\study25\\project\\1.패션.xlsx" --sheet Sheet1 --enqueue --db .\\review_bot.sqlite
  python review_full_onefile_fixed.py --reply-preview --db .\\review_bot.sqlite
  python review_full_onefile_fixed.py --report --period daily --db .\\review_bot.sqlite

  # 샘플 한 문장 추론 + 답글(모델 있으면 사용)
  python review_full_onefile_fixed.py --predict "배송이 너무 늦었어요 환불하고 싶습니다" --aspect 배송 --model .\\_save\\attrsent\\model.joblib
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from hashlib import md5

import re
import sqlite3

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import time, logging, random
try:
    import requests  # 선택: 로컬 LLM(ollama 등) 연동용
except Exception:
    requests = None

# -------------------- 설정 --------------------
# 간단 로거
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("review-bot").info
log_debug = logging.getLogger("review-bot").debug

@dataclass
class CFG:
    DB_PATH: str = "./review_bot.sqlite"
    PLATFORM_NAME: str = "aihub_attrsent"
    RANDOM_STATE: int = 42
    # --- Freeform / LLM 옵션 ---
    USE_FREEFORM: bool = False          # True면 자연어형 답글 생성
    LLM_BACKEND: str = "none"           # none|ollama
    LLM_MODEL: str = "google/gemma-3n-E4B"    # ollama 모델명 예: gemma3:latest
    MAX_REPLY_LEN: int = 300            # 답글 최대 길이(문자)
    OLLAMA_URL: str = "http://localhost:11434/api/generate"

TEXT_KEYS = ["text", "sentence", "document", "review", "contents"]
LABEL_KEYS = ["polarity", "label", "sentiment", "target_polarity", "y"]
ASPECT_KEYS = ["attribute", "aspect", "category", "target", "aspect_term"]
ID_KEYS = ["id", "review_id", "doc_id", "uid"]

POLARITY_MAP = {
    "positive": 1, "pos": 1, "긍정": 1, "1": 1, 1: 1,
    "neutral": 0, "neu": 0, "중립": 0, "0": 0, 0: 0,
    "negative": -1, "neg": -1, "부정": -1, "-1": -1, -1: -1,
}

# -------------------- 파일 읽기/정규화 --------------------

def _read_file(path: Path, sheet_name: Optional[str]=None) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".csv", ".tsv"]:
        # UTF-8 실패 시 CP949(윈도우 한글) 자동 재시도
        try:
            df = pd.read_csv(path, encoding="utf-8", sep="," if suf==".csv" else "	")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949", sep="," if suf==".csv" else "	")
    elif suf in [".jsonl", ".jl"]:
        with open(path, "r", encoding="utf-8") as f:
            df = pd.read_json(f, lines=True)
    elif suf == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                df = pd.read_json(f)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                df = pd.read_json(f, lines=True)
    elif suf == ".xlsx":
        # openpyxl 권장
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        except Exception:
            df = pd.read_excel(path, sheet_name=sheet_name)
    elif suf == ".xls":
        # xlrd 2.x는 xls 미지원 → 1.2.0 필요
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
        except Exception as e:
            raise RuntimeError(".xls 파일은 xlrd==1.2.0 설치가 필요합니다. pip install xlrd==1.2.0") from e
    else:
        raise ValueError(f"Unsupported format: {path}")
    df["__srcfile__"] = str(path)
    return df


def _auto_columns(
    df: pd.DataFrame,
    override_text: Optional[str]=None,
    override_label: Optional[str]=None,
    override_aspect: Optional[str]=None,
    override_id: Optional[str]=None,
) -> Tuple[str, Optional[str], Optional[str], str]:
    # override가 주어지면 우선 사용
    def pick(cands):
        for k in cands:
            for c in df.columns:
                if c.lower() == k:
                    return c
        return None

    text_col = override_text or pick(TEXT_KEYS) or df.columns[0]
    label_col = override_label or pick(LABEL_KEYS)
    aspect_col = override_aspect or pick(ASPECT_KEYS)
    id_col = override_id or pick(ID_KEYS) or df.columns[0]
    return text_col, label_col, aspect_col, id_col


def _normalize(
    df: pd.DataFrame,
    override_text: Optional[str]=None,
    override_label: Optional[str]=None,
    override_aspect: Optional[str]=None,
    override_id: Optional[str]=None,
) -> pd.DataFrame:
    tcol, lcol, acol, idcol = _auto_columns(df, override_text, override_label, override_aspect, override_id)
    out = pd.DataFrame()

    out["text"] = df[tcol].astype(str)
    out["aspect"] = df[acol].astype(str) if acol and acol in df.columns else None

    if lcol and lcol in df.columns:
        raw = df[lcol]
        def map_pol(x):
            if isinstance(x, str):
                x = x.strip().lower()
            return POLARITY_MAP.get(x, np.nan)
        y = raw.apply(map_pol)
        if y.isna().mean() > 0.5:
            try:
                y2 = raw.astype(int)
                uniq = sorted({int(v) for v in y2})
                lo, hi = min(uniq), max(uniq)
                mapping = {lo: -1, hi: 1}
                if len(uniq) >= 3:
                    mapping[uniq[len(uniq)//2]] = 0
                y = y2.map(mapping)
            except Exception:
                y = pd.Series([np.nan] * len(df))
        out["label"] = y
    else:
        out["label"] = np.nan

    # doc_id 생성: 지정된 id 컬럼이 없거나 text 컬럼과 같으면 해시 기반으로 생성
    if idcol and idcol in df.columns and idcol != tcol:
        out["doc_id"] = df[idcol].astype(str)
    else:
        # 파일 경로 + 인덱스 + 텍스트로 16자리 해시 ID 생성
        srcs = df.get("__srcfile__", pd.Series(["unknown"]*len(df)))
        out["doc_id"] = [md5((str(s) + "|" + str(p) + "|" + str(i)).encode("utf-8")).hexdigest()[:16]
                          for i, (s, p) in enumerate(zip(out["text"], srcs))]

    # 공백/중복 제거
    out = out[~out["text"].astype(str).str.strip().eq("")]
    out = out.drop_duplicates(subset=["doc_id"])  # 보수적
    return out.reset_index(drop=True)


def load_dataset(
    data_dir: str,
    file_glob: Optional[str]=None,
    sheet_name: Optional[str]=None,
    col_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    d = Path(data_dir)
    if not d.exists():
        raise FileNotFoundError(d)
    pats = file_glob or "**/*.*"
    files = [p for p in d.glob(pats)
             if p.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".jl", ".xlsx", ".xls"}]
    if not files:
        raise RuntimeError("No data files (csv/tsv/json/jsonl/xlsx/xls)")

    dfs = []
    for f in tqdm(files, desc="📥 Reading files", unit="file"):
        try:
            df = _read_file(f, sheet_name=sheet_name)
            dfs.append(df)
        except Exception as e:
            print(f"skip: {f} ({e})")
    raw = pd.concat(dfs, axis=0, ignore_index=True)

    col_map = col_map or {}
    norm = _normalize(
        raw,
        override_text=col_map.get("text"),
        override_label=col_map.get("label"),
        override_aspect=col_map.get("aspect"),
        override_id=col_map.get("id"),
    )
    print(f"Loaded {len(raw)} → normalized {len(norm)} (files={len(files)})")
    return norm


def load_single_file(
    file_path: str,
    sheet_name: Optional[str]=None,
    col_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = _read_file(p, sheet_name=sheet_name)
    col_map = col_map or {}
    norm = _normalize(
        df,
        override_text=col_map.get("text"),
        override_label=col_map.get("label"),
        override_aspect=col_map.get("aspect"),
        override_id=col_map.get("id"),
    )
    print(f"Loaded 1 file({p.name}) → normalized {len(norm)}")
    return norm

# -------------------- 모델 학습/평가 --------------------

def train_eval(df: pd.DataFrame, out_dir: Optional[str]=None):
    data = df.dropna(subset=["label"]).reset_index(drop=True)
    if len(data) < 100:
        print("(Not enough labeled samples to train)")
        return None
    log(f"Training on {len(data)} labeled samples …")
    X = data["text"].astype(str).tolist()
    y = data["label"].astype(int).tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=CFG.RANDOM_STATE, stratify=y)
    t0 = time.time()
    log("Vectorizing (TF-IDF char 2-4)…")
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2,4), min_df=2, max_df=0.95)
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)
    log(f"Vectorized in {time.time()-t0:.2f}s, dim={Xtr.shape[1]:,}")

    log("Training LogisticRegression …")
    t1 = time.time()
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    log(f"Model trained in {time.time()-t1:.2f}s")

    f1_macro = f1_score(y_te, pred, average='macro')
    print("\n=== Classification Report (-1,0,1) ===")
    print(classification_report(y_te, pred, digits=4))
    print(f"F1-macro: {f1_macro:.4f}")

    model_pack = {"vectorizer": vec, "model": clf}

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": X_te, "y_true": y_te, "y_pred": pred}).to_csv(Path(out_dir)/"predictions.csv", index=False)
        with open(Path(out_dir)/"report.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_te, pred, digits=4))
            f.write(f"\nF1-macro: {f1_macro:.6f}\n")
        try:
            import joblib
            joblib.dump(model_pack, Path(out_dir)/"model.joblib")
            print(f"Saved: {out_dir}/model.joblib")
        except Exception:
            print("(joblib not installed; skip model save)")
    return model_pack

# -------------------- DB/리뷰 저장 --------------------
DDL = """
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  platform TEXT NOT NULL,
  review_id TEXT NOT NULL,
  author TEXT,
  rating REAL,
  content TEXT,
  created_at TEXT,
  crawled_at TEXT NOT NULL,
  UNIQUE(platform, review_id)
);
CREATE TABLE IF NOT EXISTS replies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  platform TEXT NOT NULL,
  review_id TEXT NOT NULL,
  reply_text TEXT NOT NULL,
  replied_at TEXT,
  status TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS faq_answers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  intent TEXT NOT NULL,
  topic TEXT NOT NULL,
  answer TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    for stmt in DDL.strip().split(";\n"):
        if stmt.strip():
            conn.execute(stmt)
    conn.commit()
    conn.close()
    print(f"DB initialized: {db_path}")


def import_to_db(df: pd.DataFrame, db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    def to_star(x):
        if pd.isna(x):
            return None
        return {-1:1, 0:3, 1:5}.get(int(x), None)
    now = pd.Timestamp.utcnow().isoformat()
    n_new = 0
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="🗄️ Importing to DB"):
        try:
            conn.execute(
                "INSERT OR IGNORE INTO reviews(platform, review_id, author, rating, content, created_at, crawled_at) VALUES(?,?,?,?,?,?,?)",
                (CFG.PLATFORM_NAME, r.doc_id, None, to_star(getattr(r, "label", None)), r.text, None, now)
            )
            n_new += conn.total_changes>0
        except Exception:
            pass
    conn.commit()
    conn.close()
    print(f"Imported to DB: {n_new} rows → {db_path}")

# -------------------- 세분 자동답글(의도/심각도) --------------------
ASPECT_MAP = {
    "배송":  ["배송", "배달", "도착", "택배", "포장", "수령", "지연"],
    "품질":  ["품질", "재질", "내구", "원단", "마감", "상태", "불량", "파손"],
    "가격":  ["가격", "비싸", "저렴", "가성비", "할인", "프로모션"],
    "사이즈": ["사이즈", "크기", "핏", "길이", "넉넉", "타이트"],
    "CS":   ["응대", "고객센터", "교환", "환불", "상담", "불친절"],
}

def _aspect_group(a: Optional[str]) -> str:
    if not a or a == "None":
        return "기타"
    a_low = str(a).lower()
    for k, vocab in ASPECT_MAP.items():
        for v in vocab:
            if v.lower() in a_low:
                return k
    return "기타"

INTENT_PATTERNS = {
    "refund":        r"환불|refund",
    "exchange":      r"교환|교체|교환해",
    "late_delivery": r"지연|늦|배송.*늦|배송.*지연",
    "damaged":       r"파손|깨졌|찢어|스크래치|불량",
    "missing":       r"누락|없었|빠졌",
    "rude":          r"불친절|무례|태도",
    "question":      r"\?|문의|가능하나요|알고 싶|어떻게",
    "feature":       r"추가|기능|되면 좋|개선|요청",
}

def detect_intents(text: str) -> set:
    t = (text or "").lower()
    tags = set()
    for name, pat in INTENT_PATTERNS.items():
        if re.search(pat, t, flags=re.I):
            tags.add(name)
    return tags

KW_VERY_NEG = ["최악", "다시는", "환불", "법적", "고발", "사기", "불량", "파손", "누락", "분노"]
KW_VERY_POS = ["최고", "완벽", "강추", "최애", "대만족", "완전 좋", "인생템", "감동"]

def estimate_severity(base_label: Optional[int], text: str) -> int:
    sev = 0 if base_label is None else int(base_label)
    tl = (text or "").lower()
    if any(k in tl for k in KW_VERY_NEG):
        sev = min(sev, -1) - 1  # down to -2
    if any(k in tl for k in KW_VERY_POS):
        sev = max(sev,  1) + 1  # up to +2
    return max(min(sev, 2), -2)

REPLY_TEMPLATES = {
    ("배송", -2): "배송 문제로 큰 불편을 드려 진심으로 사과드립니다. 주문번호를 알려주시면 즉시 조사하고 필요한 조치를 진행하겠습니다.",
    ("배송", -1): "배송 지연으로 불편을 드려 죄송합니다. 더 정확한 배송 안내와 처리 속도를 개선하겠습니다.",
    ("배송",  0): "배송 관련 의견 감사합니다. 보다 안정적인 배송을 위해 계속 점검하겠습니다.",
    ("배송",  1): "배송이 만족스러우셨다니 기쁩니다. 다음에도 빠르고 안전하게 전달드리겠습니다!",
    ("배송",  2): "배송이 매우 만족스러우셨다니 감사합니다. 늘 같은 경험을 드리겠습니다!",

    ("품질", -2): "제품 문제로 큰 실망을 드렸습니다. 불량/파손 사진을 보내주시면 즉시 교환·환불을 도와드리겠습니다.",
    ("품질", -1): "품질 이슈를 주셔서 감사합니다. 해당 로트를 점검하고 재발 방지에 반영하겠습니다.",
    ("품질",  0): "품질 관련 의견 감사합니다. 더 나은 기준을 위해 꾸준히 개선하겠습니다.",
    ("품질",  1): "품질을 좋게 평가해주셔서 감사합니다. 기대에 부응하겠습니다!",
    ("품질",  2): "최고의 평가에 감사드립니다. 품질로 계속 만족 드리겠습니다!",

    ("가격", -2): "가격으로 크게 실망을 드려 죄송합니다. 프로모션/혜택을 더 알기 쉽게 안내드리겠습니다.",
    ("가격", -1): "가격에 대한 의견 감사드리며, 합리적 가격 정책을 위해 검토하겠습니다.",
    ("가격",  0): "가격 관련 의견 감사합니다. 더 나은 가치를 드리겠습니다.",
    ("가격",  1): "가격 만족 리뷰 감사합니다. 가성비 좋은 상품으로 보답하겠습니다!",
    ("가격",  2): "최고의 가성비 평가 감사합니다. 좋은 품질을 합리적인 가격으로 제공하겠습니다!",

    ("사이즈", -2): "사이즈로 큰 불편을 드려 죄송합니다. 교환 절차와 상세 사이즈 상담을 도와드리겠습니다.",
    ("사이즈", -1): "사이즈 의견 감사드립니다. 상세 가이드를 더 정확히 안내하겠습니다.",
    ("사이즈",  0): "사이즈 관련 의견 감사합니다. 표기 개선에 반영하겠습니다.",
    ("사이즈",  1): "사이즈가 잘 맞으셨다니 다행입니다!",
    ("사이즈",  2): "최상의 착용감 평가 감사드립니다. 같은 핏으로 또 찾아뵐게요!",

    ("CS", -2): "응대 과정에서 큰 불편을 드려 진심으로 사과드립니다. 관련 내용을 즉시 시정하고 책임 있게 조치하겠습니다.",
    ("CS", -1): "응대 관련 불편을 드려 죄송합니다. 교육·프로세스를 개선하겠습니다.",
    ("CS",  0): "응대 관련 의견 감사합니다. 더 신속하고 정확히 도와드리겠습니다.",
    ("CS",  1): "만족스러운 상담이었다니 감사드립니다.",
    ("CS",  2): "최고의 응대 평가에 감사드립니다. 항상 같은 경험을 드리겠습니다!",

    ("기타", -2): "큰 불편을 드려 죄송합니다. 상세 상황을 알려주시면 즉시 해결하겠습니다.",
    ("기타", -1): "불편을 드려 죄송합니다. 더 나은 경험을 위해 개선하겠습니다.",
    ("기타",  0): "소중한 의견 감사합니다. 개선에 반영하겠습니다.",
    ("기타",  1): "좋은 리뷰 감사합니다. 더 나은 경험을 제공하겠습니다.",
    ("기타",  2): "최고의 평가 감사합니다! 기대에 더 크게 보답하겠습니다.",
}

INTENT_OVERLAYS = [
    ("refund",        "환불을 요청해 주셔서 확인 즉시 진행하겠습니다. 주문번호와 결제 정보를 DM/메일로 공유 부탁드립니다."),
    ("exchange",      "교환 요청 건 확인했습니다. 절차와 반송 라벨을 안내드리겠습니다."),
    ("late_delivery", "배송 지연에 대해 사과드립니다. 현재 위치를 확인해 즉시 안내 드리겠습니다."),
    ("damaged",       "파손/불량으로 불편을 드려 죄송합니다. 사진을 보내주시면 교환/환불을 빠르게 처리하겠습니다."),
    ("missing",       "구성품 누락 건 확인 즉시 보완 발송하겠습니다. 부족한 항목을 알려주세요."),
    ("rude",          "응대 과정에서 불편을 드려 죄송합니다. 교육과 모니터링을 강화하겠습니다."),
    ("question",      "문의 주신 사항에 대해 자세히 안내드리겠습니다. 추가로 궁금하신 점도 편히 말씀해주세요."),
    ("feature",       "소중한 제안 감사합니다. 관련 부서와 공유하여 검토하겠습니다."),
]

# -------------------- 키워드 추출 & 자유형 답글 생성 --------------------
_KO_STOP = set(["그리고","그러나","하지만","또한","또","그","이","저","것","수","등","및","에서","으로","부터","까지","에게","께","은","는","이","가","을","를","에","의","도","만","보다","처럼","하다","되다","있다","없다","같다","거","좀","너무"])

def _simple_tokens(s: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", (s or "").lower())
    return [t for t in toks if t not in _KO_STOP]

def extract_keywords(text: str, topk: int = 6) -> list:
    toks = _simple_tokens(text)
    if not toks:
        return []
    from collections import Counter
    uni = Counter(toks)
    bi = Counter([f"{a} {b}" for a, b in zip(toks, toks[1:])])
    cand = []
    cand.extend([(w, c) for w, c in uni.items()])
    cand.extend([(w, c*1.5) for w, c in bi.items()])
    cand = sorted(cand, key=lambda x: x[1], reverse=True)
    out = []
    for w, _ in cand:
        if all(w not in o and o not in w for o in out):
            out.append(w)
        if len(out) >= topk:
            break
    return out

def _truncate(s: str, n: int):
    return s if len(s) <= n else s[:n-1] + "…"

def _llm_generate(prompt: str) -> Optional[str]:
    if CFG.LLM_BACKEND != "ollama":
        return None
    if requests is None:
        return None
    try:
        resp = requests.post(CFG.OLLAMA_URL, json={"model": CFG.LLM_MODEL, "prompt": prompt, "stream": False}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            txt = data.get("response") or data.get("text")
            return txt
    except Exception:
        return None
    return None

_OPENERS = [
    "안녕하세요, 리뷰 남겨주셔서 감사합니다.",
    "소중한 의견 고맙습니다.",
    "시간 내어 후기 주셔서 감사해요.",
]
_APOLOGIES = [
    "불편을 드려 정말 죄송합니다.",
    "아쉬운 경험을 하시게 해 죄송합니다.",
]
_POSITIVE = [
    "만족하셨다니 저희도 무척 기쁩니다!",
    "좋은 경험을 하셨다니 감사합니다.",
]
_CLOSERS = [
    "더 나은 경험으로 보답하겠습니다.",
    "항상 같은 품질로 찾아뵙겠습니다.",
    "추가로 궁금하신 점은 언제든 말씀해주세요.",
]

def compose_reply_freeform(aspect_group: str, severity: int, intents: set, text: str) -> str:
    kws = extract_keywords(text, topk=6)
    kw_line = ", ".join(kws[:4]) if kws else None
    prompt = f"""
당신은 고객 리뷰에 답글을 작성하는 CS 담당자입니다. 한국어로 정중하고 자연스럽게 2~4문장의 답글을 쓰세요.
조건:
- 리뷰 핵심 키워드: {kws}
- 속성 그룹: {aspect_group}
- 심각도: {severity} (-2 매우 부정 ~ +2 매우 긍정)
- 의도 태그: {sorted(list(intents))}
- 금지: 과도한 형식문구 반복, 이모지 남용, 과장 표현
- 길이: {CFG.MAX_REPLY_LEN}자 이내
답글만 출력하세요.
""".strip()
    llm_txt = _llm_generate(prompt)
    if llm_txt:
        return _truncate(llm_txt.strip(), CFG.MAX_REPLY_LEN)
    parts = []
    parts.append(random.choice(_OPENERS))
    if severity <= -1:
        parts.append(random.choice(_APOLOGIES))
    if kw_line:
        parts.append(f"말씀해 주신 ‘{kw_line}’ 부분은 내부에 공유하여 바로 점검하겠습니다.")
    overlays = []
    for key, text_tmpl in INTENT_OVERLAYS:
        if key in intents:
            overlays.append(text_tmpl)
        if len(overlays) >= 2:
            break
    if overlays:
        parts.append(" ".join(overlays))
    aspect_hint = {
        "배송": "배송 단계별 모니터링을 강화하겠습니다.",
        "품질": "해당 로트와 검수 프로세스를 재점검 중입니다.",
        "가격": "행사/혜택 안내를 더 명확히 하겠습니다.",
        "사이즈": "상세 치수/핏 가이드도 함께 개선하겠습니다.",
        "CS": "응대 품질 모니터링과 교육을 강화하겠습니다.",
        "기타": "전달 주신 의견은 즉시 개선 항목에 반영하겠습니다.",
    }.get(aspect_group, "의견은 개선 항목에 반영하겠습니다.")
    parts.append(aspect_hint)
    parts.append(random.choice(_CLOSERS))
    reply = " ".join(parts)
    return _truncate(reply, CFG.MAX_REPLY_LEN)

def compose_reply(aspect_group: str, severity: int, intents: set, text_for_freeform: str = "") -> str:
    # ▶ Freeform 모드면 자연어형으로 생성 (Gemma3/백업 규칙형)
    if getattr(CFG, "USE_FREEFORM", False):
        return compose_reply_freeform(aspect_group, severity, intents, text_for_freeform)

    # ▶ 기존 템플릿 모드
    base = REPLY_TEMPLATES.get((aspect_group, severity)) \
        or REPLY_TEMPLATES.get((aspect_group, 0)) \
        or REPLY_TEMPLATES.get(("기타", 0))
    overlays = []
    for key, text in INTENT_OVERLAYS:
        if key in intents:
            overlays.append(text)
        if len(overlays) >= 2:
            break
    msg = base
    if overlays:
        msg += "\n" + "\n".join(overlays)
    return msg


def enqueue_replies_from_df(df: pd.DataFrame, db_path: str):
    conn = sqlite3.connect(db_path)
    cnt = 0
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="✉️ Enqueue replies"):
        doc_id = r.doc_id
        text   = getattr(r, "text", "") or ""
        base_label = None if pd.isna(getattr(r, "label", None)) else int(getattr(r, "label"))
        aspect_g   = _aspect_group(getattr(r, "aspect", None))
        severity   = estimate_severity(base_label, text)  # -2..+2
        intents    = detect_intents(text)
        reply      = compose_reply(aspect_g, severity, intents, text)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO replies(platform, review_id, reply_text, status) VALUES(?,?,?,?)",
                (CFG.PLATFORM_NAME, doc_id, reply, "pending")
            )
            cnt += conn.total_changes > 0
        except Exception:
            pass
    conn.commit()
    conn.close()
    print(f"Enqueued replies: {cnt}")


def reply_preview(db_path: str, limit: int=10):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT platform, review_id, reply_text FROM replies WHERE status='pending' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    if not rows:
        print("(No pending replies)")
        return
    print(f"\n===== Reply Preview (latest {len(rows)}) =====")
    for i, r in enumerate(rows, 1):
        print(f"[{i}] ({r[0]}) review_id={r[1]}\n→ {r[2]}\n")

# -------------------- 콘솔 리포트 --------------------
from datetime import datetime, timedelta
try:
    from dateutil import tz
    SEOUL_TZ = tz.gettz("Asia/Seoul")
except Exception:
    SEOUL_TZ = None

def _period(period: str) -> Tuple[str,str]:
    now = datetime.now(SEOUL_TZ) if SEOUL_TZ else datetime.now()
    if period=="daily":
        s = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        e = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    elif period=="weekly":
        wd = now.weekday()
        e = (now - timedelta(days=wd+1)).replace(hour=23, minute=59, second=59, microsecond=0)
        s = e - timedelta(days=6)
    elif period=="monthly":
        first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        e = first_this - timedelta(seconds=1)
        s = (first_this - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError("period must be daily/weekly/monthly")
    return s.isoformat(), e.isoformat()


def report_stdout(db_path: str, period: str="daily", topn: int=10):
    s, e = _period(period)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT platform, author, rating, content, created_at, crawled_at FROM reviews WHERE crawled_at BETWEEN ? AND ? ORDER BY crawled_at DESC",
        conn, params=(s, e)
    )
    conn.close()
    total = len(df)
    avg_rating = round(float(df["rating"].dropna().mean()), 3) if total else 0.0
    print("\n==============================")
    print(f"📊 Review Summary ({period})")
    print(f"기간: {s} ~ {e}")
    print("------------------------------")
    print(f"총 리뷰 수   : {total}")
    print(f"평균 평점    : {avg_rating if total else '-'}")
    print("------------------------------")
    if total:
        print(f"📝 최근 리뷰 Top {min(topn, total)}")
        for i, row in enumerate(df.head(topn).itertuples(index=False), 1):
            content = (row.content or "").strip().replace("\n"," ")
            if len(content) > 120:
                content = content[:120] + "…"
            print(f"[{i}] ({row.platform}) ★{row.rating or '-'} | {row.created_at or row.crawled_at}")
            print(f"     {content}")
    else:
        print("해당 기간 리뷰가 없습니다.")

# -------------------- 샘플 추론(문장 → 감성/답글) --------------------

def _load_model(model_path: Optional[str]):
    if not model_path:
        return None
    p = Path(model_path)
    if not p.exists():
        print(f"(model not found: {model_path})")
        return None
    try:
        import joblib
        return joblib.load(p)
    except Exception as e:
        print(f"(failed to load model: {e})")
        return None


def predict_and_reply(text: str, aspect_hint: Optional[str], model_path: Optional[str]):
    pack = _load_model(model_path)
    if pack:
        vec, mdl = pack["vectorizer"], pack["model"]
        X = vec.transform([text])
        base_label = int(mdl.predict(X)[0])  # -1/0/1
    else:
        # 간단 휴리스틱
        t = text.lower()
        score = 0
        for k in ["좋", "만족", "추천", "최고", "감동", "great", "love"]:
            if k in t: score += 1
        for k in ["불만", "환불", "최악", "파손", "늦", "지연", "별로", "bad"]:
            if k in t: score -= 1
        base_label = 1 if score>0 else (-1 if score<0 else 0)

    sev = estimate_severity(base_label, text)
    asp_g = _aspect_group(aspect_hint or "기타")
    intents = detect_intents(text)
    reply = compose_reply(asp_g, sev, intents, text)

    print("문장:", text)
    print("추정 감성(-2..+2):", sev, "/ 기본라벨:", base_label)
    print("의도태그:", ", ".join(sorted(intents)) or "(없음)")
    print("--- 자동답글 ---\n" + reply)

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FULL One-file Review Baseline (FIXED)")

    # 로그 레벨
    ap.add_argument("--verbose", action="store_true", help="자세한 진행 로그")
    ap.add_argument("--quiet", action="store_true", help="진행 로그 최소화")

    # 데이터 입력 방식
    ap.add_argument("--data", type=str, default=None, help="디렉토리(재귀) 경로")
    ap.add_argument("--file", type=str, default=None, help="단일 파일 경로 (csv/tsv/json/jsonl/xlsx/xls)")
    ap.add_argument("--glob", type=str, default=None, help="파일 패턴 (예: **/train*.jsonl)")
    ap.add_argument("--sheet", type=str, default=None, help="엑셀 시트명")

    # 컬럼 수동 매핑(선택)
    ap.add_argument("--text-col", type=str, default=None)
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--aspect-col", type=str, default=None)
    ap.add_argument("--id-col", type=str, default=None)

    # DB & 파이프라인
    ap.add_argument("--init-db", action="store_true")
    ap.add_argument("--db", type=str, default=CFG.DB_PATH)

    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", type=str, default=None)

    ap.add_argument("--import-to-db", action="store_true")

    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--reply-preview", action="store_true")

    ap.add_argument("--report", action="store_true")
    ap.add_argument("--period", choices=["daily","weekly","monthly"], default="daily")

    ap.add_argument("--predict", type=str, default=None, help="한 문장 평가/답글 생성")
    ap.add_argument("--aspect", type=str, default=None, help="문장 관련 속성 힌트(예: 배송/품질/가격/사이즈/CS)")
    ap.add_argument("--model", type=str, default=None, help="model.joblib 경로(선택)")

    # 자유형/LLM 옵션
    ap.add_argument("--freeform", action="store_true", help="자연어형 답글 생성(템플릿 대신)")
    ap.add_argument("--llm", choices=["none","ollama"], default="none", help="LLM 백엔드 선택")
    ap.add_argument("--llm-model", type=str, default="google/gemma-3n-E4B", help="LLM 모델명(ollama)")
    ap.add_argument("--max-reply-len", type=int, default=300, help="답글 최대 길이")

    args = ap.parse_args()

    # Freeform/LLM 설정 반영
    CFG.USE_FREEFORM = bool(args.freeform)
    CFG.LLM_BACKEND = args.llm
    CFG.LLM_MODEL = args.llm_model
    CFG.MAX_REPLY_LEN = int(args.max_reply_len)

    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)
    log("🚀 Start")

    # 컬럼 매핑 dict 구성
    col_map = {
        k: v for k, v in {
            "text": args.text_col,
            "label": args.label_col,
            "aspect": args.aspect_col,
            "id": args.id_col,
        }.items() if v
    }

    if args.init_db:
        init_db(args.db)

    df = None
    if args.file:
        df = load_single_file(args.file, sheet_name=args.sheet, col_map=col_map)
        if args.preview:
            print(df.head(5))
    elif args.data:
        df = load_dataset(args.data, file_glob=args.glob, sheet_name=args.sheet, col_map=col_map)
        if args.preview:
            print(df.head(5))

    if (args.train or args.eval) and df is not None:
        _ = train_eval(df, args.out)

    if args.import_to_db and df is not None:
        import_to_db(df, args.db)

    if args.enqueue:
        if df is None:
            print("(--enqueue 사용 시 --file 또는 --data 로 데이터를 먼저 로드하세요)")
        else:
            enqueue_replies_from_df(df, args.db)

    if args.reply_preview:
        reply_preview(args.db)

    if args.report:
        report_stdout(args.db, args.period)

    if args.predict:
        predict_and_reply(args.predict, args.aspect, args.model)
