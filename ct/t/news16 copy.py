#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import sys, re, datetime, json, hashlib, os, random, warnings
from typing import Optional, List, Callable

import numpy as np
import pandas as pd
from collections import Counter
from optuna.exceptions import TrialPruned
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False

import optuna
from optuna.exceptions import TrialPruned

# ===== Embedding backends (bge / e5 / SimCSE) =====
try:
    from FlagEmbedding import BGEM3FlagModel  # bge-m3
    HAS_BGE = True
except Exception:
    HAS_BGE = False

try:
    from sentence_transformers import SentenceTransformer  # e5 / ko-SimCSE
    HAS_ST = True
except Exception:
    HAS_ST = False

# ---------- 1) Config ----------
class Config:
    PATH = './ct/t'
    SAVE_PATH = './ct/t'
    CSV_NAME = "samples.csv"
    RANDOM_STATE = 193
    OPTUNA_TRIALS = 300
    CHAR_LIMIT = 1200

    # Embedding rerank
    USE_EMB_RERANK = True
    EMB_BACKEND = "e5"          # "auto" | "bge" | "e5" | "sbert" | "none"
    EMB_MODEL = None
    EMB_WEIGHT = 0.39
    EMB_BATCH = 32

    # Optuna / infra
    USE_PRUNER = True
    STORAGE = "sqlite:///optuna_prompt.db"      # 영구 저장
    STUDY_NAME = "prompt_meta_optimization"
    N_STARTUP = 20                               # TPE warmup
    MULTIVARIATE = True                          # 파라미터 상호작용
    CONSTANT_LIAR = True                         # 병렬 안전(미지원 버전 대비 try/except 처리)
    N_SPLITS = 5
    SPLIT_CACHE = "./ct/t/splits.json"           # 고정 스플릿 저장
    WARM_START_FILE = "./ct/t/best_params.json"  # 지난 베스트 파라미터

    # Ray Tune 프리패스
    USE_RAY_PRIME = True          # 1차 거르기 활성화
    RAY_SCHEDULER = "ASHA"        # "ASHA" | "BOHB"
    RAY_PRIME_TIME_S = 120
    RAY_PRIME_NUM_SAMPLES = 30
    RAY_PRIME_TOPK = 5
    FORCE_REBUILD_SPLITS = False
    # 힌트 노출 정책
    HINT_BUDGET = 80
    FORCE_BUDGET_POS = 20
    FORCE_BUDGET_NEG = 20
    VARIANT_POLICY = "after"      # "none" | "after" | "learned_only"
    VARIANT_PER_TOKEN = 1
    USE_HINT_MMR = True

    # --- negative hint policy ---
    MIN_NEG_HINTS = 20           # 최종 프롬프트에 '제외' 힌트를 최소 이 개수 이상 보장
    NEG_FORCE_RATIO = 0.20       # k의 20%는 강제 제외 힌트에 할당
    NEG_LEARNED_MIN_RATIO = 0.60 # k의 60%는 "학습된" 제외 힌트로 보장

cfg = Config()
os.makedirs(cfg.SAVE_PATH, exist_ok=True)

# ---------- 1.1) 고정 서치스페이스(동적 방지의 단일 출처) ----------
SEARCH_SPACE = {
    "TOP_N":        ("int",   200, 500, 2),
    "MIN_DF":       ("int",     2,   6, 1),
    "NGRAM_MAX":    ("int",     2,   5, 1),
    "NEUTRAL_BAND": ("float", 0.07, 0.40, None),
    "HINT_K":       ("int",    45,  70, 1),
}
SPACE_SIG = hashlib.sha1(json.dumps(SEARCH_SPACE, sort_keys=True).encode()).hexdigest()[:8]
PROPER_WHITELIST: set[str] = set()

def _suggest_from_space(trial: optuna.Trial, name: str):
    kind, lo, hi, step = SEARCH_SPACE[name]
    if kind == "int":
        return trial.suggest_int(name, lo, hi, step=step or 1)
    if kind == "float":
        return trial.suggest_float(name, lo, hi)
    raise ValueError(kind)

def _coerce_to_search_space(params: dict) -> Optional[dict]:
    """모든 키가 존재하고, 범위/스텝 일치할 때만 정제된 dict 반환. 아니면 None."""
    fixed = {}
    for k, spec in SEARCH_SPACE.items():
        kind, lo, hi, step = spec
        if k not in params:
            return None
        v = params[k]
        if kind == "int":
            try:
                iv = int(v)
            except Exception:
                return None
            if not (lo <= iv <= hi):
                return None
            if step and ((iv - lo) % step != 0):
                return None
            fixed[k] = iv
        elif kind == "float":
            try:
                fv = float(v)
            except Exception:
                return None
            if not (lo <= fv <= hi):
                return None
            fixed[k] = fv
        else:
            return None
    return fixed

# ---------- 2) Globals & utils ----------

# === (REPLACE) 노이즈 ===
NOISE_TOKENS = {
    "것","등","및","등등","분야","측","관계자","이번","자체","강화","중요","중요한",
    "시장","경쟁력","안정적","확대","추진","계획",
    "밝혔다","전했다","보였다","보인다","있다","최대","최근","다시","세계","중심","정부","우리","우리의",
    "있어","이상","이상의","관계자는이번","관계자","세계","중심","하","있다특히","국","시장",
    "따라","다른","대규모","주목된다","함께","높","제공하","강력한"
}

# === (REPLACE) 모호어(근접 규칙 전용) ===
AMBIG_TOKENS = {
    "배터리","bms","셀","팩","모듈",
    "반도체","칩","soc","mcu","npu",
    "ai","인공지능","알고리즘","데이터","플랫폼","클라우드","에너지"
}

# --- (REPLACE) 자동차 앵커(근접/게이트에서 사용) ---
AUTO_ANCHORS = [
    "현대차","기아","테슬라","BMW","벤츠","메르세데스","GM","포드","토요타","폭스바겐","아우디","르노","볼보",
    "아이오닉","EV6","코나 일렉트릭","모델3","모델Y",
    "전기차","수소차","하이브리드","자율주행","ADAS","SDV","오토파일럿","FSD","레벨 4","L4",
    "충전소","충전기","V2G","V2L","차량용 반도체","OTA","커넥티드카","로보택시","양산","리콜","공장"
]
AUTO_POS_THR = 0.35
AUTO_NEG_THR = 0.30
AUTO_ANCHORS_N = [ re.sub(r"\s+","", a.lower()) for a in AUTO_ANCHORS ]  # ← 이 줄은 그대로 유지/재사용

# --- (REPLACE) 포함 강제(보조) ---
FORCE_POS_EXTRA = [
    "자동차","완성차","전기차","자율 주행","충전","충전 인프라","보조금",
    "생산","양산","공장","라인","증설","리콜","인증","안전기준",
    "출시","판매","출고","수출","실적","점유율",
    "합작법인","JV","JDA","MOU","상용화","시범운행","파일럿","인포테인먼트","OTA","커넥티드카"
]

# --- (REPLACE) 강한 제외(다른 산업) ---
FORCE_NEG_EXTRA = [
    "철도","KTX","항공","항공기","공항","보잉","에어버스",
    "선박","조선","해운","해양플랜트",
    "UAM","드론택시",
    "ESS","태양광","풍력","원전","원자력",
    "방산","미사일","레이더"
]

# --- (ADD) 소프트 제외(앵커 없을 때만 제외 판단) ---
SOFT_NEG_TOKENS = [
    "관세","무역전쟁","보호무역",
    "환율","금리","CPI","증시","코스피","나스닥",
    "트럼프","대선","선거","의회","법안",
    "경제","경기","물가","고용"
]

# ▼ (REPLACE) 앵커 화이트리스트
PROPER_WHITELIST.update({
    "현대차","기아","테슬라","bmw","벤츠","메르세데스","gm","포드","토요타","폭스바겐","아우디","르노","볼보",
    "ev","전기차","수소차","하이브리드","adas","sdv","레벨 4","l4","fsd","오토파일럿",
    "아이오닉","ev6","코나일렉트릭","모델3","모델y",
    "충전소","충전기","v2g","v2l","양산","리콜","공장","차량용반도체","ota","커넥티드카","로보택시"
})

AI_BAN_STEMS = {"ai","인공지능","국가ai"}

# === (NEW) 형태 기반 노이즈 패턴
NOISE_REGEX = [
    # 과거시 서술어/보도 관용구(밝혔다/전했다/…했다/…혔다 등) 제거
    re.compile(r"(?:했다|하였다|되었다|됐다|전했다|밝혔다|나타났다|드러났다|알렸다)$"),
    re.compile(r"(?:되며|하며|한다|된다|있다|보인다)$"),
    re.compile(r"^[가-힣]{1,2}$"),   # 1~2글자 일반 토큰 컷 (예: '것')
    re.compile(r"^.{,1}$")
]

def _is_noise_token(t: str) -> bool:
    n = _norm_token(t)
    if n in NOISE_TOKENS:
        return True
    return any(rx.search(n) for rx in NOISE_REGEX)

def _is_ambig_token(t: str) -> bool:
    return _norm_token(t) in AMBIG_TOKENS


VARIANT_PAIRS = []
TOKEN_PATTERNS = {
    "A": r"(?u)\b[가-힣A-Za-z]{2,}\b",
    "C": r"(?u)\b[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z]{2,}\b",
    "AN": r"(?u)(?:\b[가-힣A-Za-z]{2,}\b|\b\d{1,3}%\b|\b\d{4}\b|\b\d+(?:\.\d+)?\b)"
}
_JOSA = ("을","를","이","가","은","는","의","에","에서","으로","로","과","와","도","만","까지","부터")
PROPER_NOUN_HEUR = re.compile(r"[A-Za-z0-9]|(닷|넷|랩스|모터스|테크)$")
_DISPLAY_BACK = {"소프트웨어중심": "소프트웨어 중심"}
_DISPLAY_SUFFIXES = ("자율주행","주행","차량","용량","반도체")

# COMPACT_CORE_TMPL = ("""역할: 한국어 자동차 기사 0/1 분류기
# 목표: 기사가 '자율주행차' 및 관련 자동차 산업/기술에 관한 것인지 정확히 분류.
# 출력 규칙: 다른 설명 없이, 오직 '0'(관련 없음) 또는 '1'(관련 있음) 한 글자만 출력.
# 결정 절차:
# 1. '제외 기준'에 하나라도 명확히 해당하면 즉시 '0'으로 판정.
# 2. 1번에 해당하지 않고, '포함 기준'에 하나라도 명확히 해당하면 '1'로 판정.
# 3. 위 두 경우에 모두 해당하지 않으면 최종적으로 '0'으로 판정.
# 포함 기준 (다음 중 하나 이상 충족):
# - 명시적 주제: 기사에 완성차 브랜드, 특정 차명, 전기차(ev), 자율주행, ADAS, SDV 등이 명확히 언급됨.
# - 산업 활동: 생산, 판매, 실적, 투자, 공장, 양산, 리콜, 규제 등 자동차 산업과 직접 관련된 활동이 언급됨.
# - 모호어 맥락: '배터리', 'AI' 등의 단어가 자동차 앵커 단어(예: 현대차, 전기차, 공장 등)와 가까운 문맥에서 함께 사용됨.
# 제외 기준 (다음 중 하나 이상 충족):
# - 다른 산업: 철도, 항공, 선박, UAM, ESS, 태양광, 방산 등 자동차가 아닌 다른 산업이 주된 내용임.
# - 무관한 내용: 은유적 표현, 단순 교통사고, 적용 분야가 불명확한 일반 기술(반도체, AI 등)만 단독으로 언급됨.""")

COMPACT_CORE_TMPL = ("""역할: 한국어 자동차 기사 0/1 분류기
목표: 기사가 '자율주행차' 및 관련 자동차 산업/기술에 관한 것인지 정확히 분류.
출력 규칙: 다른 설명 없이, 오직 '0'(관련 없음) 또는 '1'(관련 있음) 한 글자만 출력.
결정 절차:
1. '제외 기준'에 하나라도 명확히 해당하면 즉시 '0'으로 판정.
2. 1번에 해당하지 않고, '포함 기준'에 하나라도 명확히 해당하면 '1'으로 판정.
3. 위 두 경우에 모두 해당하지 않으면 최종적으로 '0'으로 판정.
포함 기준 (다음 중 하나 이상 충족):
- 명시적 주제: 기사에 완성차 브랜드, 특정 차명, 전기차(ev), 자율주행, ADAS, SDV 등이 명확히 언급됨.
- 산업 활동: 생산, 판매, 실적, 투자, 공장, 양산, 리콜, 규제 등 자동차 산업과 직접 관련된 활동이 언급됨.
- 모호어 맥락: '배터리', 'AI' 등의 단어가 자동차 앵커 단어(예: 현대차, 전기차, 공장 등)와 가까운 문맥에서 함께 사용됨.
제외 기준 (다음 중 하나 이상 충족):
- 다른 산업: 철도, 항공, 선박, UAM, ESS, 태양광, 방산 등 자동차가 아닌 다른 산업이 주된 내용임.
- 무관한 내용: 은유적 표현, 단순 교통사고, 적용 분야가 불명확한 일반 기술(반도체, AI 등)만 단독으로 언급됨.
- (소프트 제외) 관세/정치/거시 키워드는 자동차 앵커 단어 없이 단독으로 등장할 때만 제외로 판단.""")


FORCE_POS_EXTRA = [
    "자동차","자율 주행","충전","중요한","차량 용","완성차",
    "전동화","현대차","우리","우리의","현지","판매","ap"
]
FORCE_NEG_EXTRA = [
    "ess","트럼프","관세","태양광","상호관세","국가 ai","방산","지능형","차기","경제"
]

PROPER_WHITELIST |= {
    "현대차","기아","테슬라","bmw","벤츠","gm","포드",
    "ev","전기차","adas","sdv","레벨 4","l4","ap","오토파일럿","충전","양산","리콜","공장"
}

MOHO_STOPWORDS = {"우리","우리의","이번","자체","높","기존","중요한","시장","시장의","가격","가격과","안정적인"}

def _arr_sig(y: np.ndarray, groups: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(y, dtype=np.int64).tobytes())
    h.update(np.asarray(groups, dtype=np.int64).tobytes())
    return h.hexdigest()

def _norm_token(t: str) -> str:
    return re.sub(r"\s+", "", str(t).lower())

def _split_josa(t: str) -> tuple[str, str]:
    s = str(t)
    for j in _JOSA:
        if s.endswith(j) and len(s) > len(j):
            return s[:-len(j)], j
    return s, ""

def _alias_key(t: str) -> str:
    stem = _norm_token(_split_josa(t)[0])   # 조사 제거한 어간
    # 인공지능 계열은 alias를 'ai'로 통합
    if ('인공지능' in stem) or stem.endswith('ai'):
        return 'ai'
    return stem

def _dedup_by_alias(lst: list[str]) -> list[str]:
    """동의어/별칭(L4/레벨4, 인공지능/AI 등)을 동일 키로 묶어 중복 제거."""
    seen, out = set(), []
    for w in lst:
        k = _alias_key(w)
        if k in seen:
            continue
        seen.add(k); out.append(w)
    return out

def _dedup_preserve_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _to_spaced_form(base: str) -> str:
    if " " in base: return base
    if base in _DISPLAY_BACK: return _DISPLAY_BACK[base]
    if re.fullmatch(r"[가-힣]{5,}", base):
        for suf in _DISPLAY_SUFFIXES:
            if base.endswith(suf) and len(base) > len(suf)+1:
                return base[:-len(suf)] + " " + suf
    return base

def _l2_normalize(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.clip(n, 1e-12, None)

def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T)

def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F\u200B-\u200F\uFEFF]", "", s)
    s = "\n".join(re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return canonize_variants(s)

def _vocab_from_vectorizer(vect):
    try: return np.asarray(vect.get_feature_names_out())
    except AttributeError: return np.asarray(vect.get_feature_names())

def build_variant_pairs_from_data(df: pd.DataFrame):
    global VARIANT_PAIRS
    _patterns = [
        ("자율주행차", "자율주행"), ("커넥티드카", "커넥티드"),
        ("테슬라", "테슬라"), ("현대자동차", "현대차"), ("기아자동차", "기아차"),
        ("모빌리티", "모빌리티"), ("반도체", "반도체")
    ]
    VARIANT_PAIRS = [(re.compile(p), r) for p, r in _patterns]

def canonize_variants(s: str) -> str:
    global VARIANT_PAIRS
    for pat, rep in VARIANT_PAIRS:
        s = pat.sub(rep, s)
    return s

def _filter_moho(tokens):
    return [t for t in tokens if _norm_token(t) not in MOHO_STOPWORDS]

def _hash_token_vecs(tokens: list[str], dim: int = 64) -> np.ndarray:
    V = np.empty((len(tokens), dim), dtype=np.float32)
    for i, t in enumerate(tokens):
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        bits = [(1.0 if ((h >> b) & 1) else -1.0) for b in range(dim)]
        v = np.asarray(bits, dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-8)
        V[i] = v
    return V

def _tiny_hash_jitter(tokens: list[str], key: str, scale: float = 1e-3) -> np.ndarray:
    out = []
    for t in tokens:
        h = int(hashlib.sha1((key + "|" + t).encode("utf-8")).hexdigest(), 16) % 1_000_000
        out.append(scale * ((h / 500_000.0) - 1.0))  # [-scale, +scale)
    return np.asarray(out, dtype=np.float32)

def _cooc_rate_bulk(texts: list[str], tokens: list[str], anchors: list[str]) -> dict[str, float]:
    """토큰이 포함된 문장에서 자동차 앵커 단어와의 동시등장 비율을 계산."""
    rates: dict[str, float] = {}
    if not tokens:
        return rates
    # 문장 단위 분할(한글/영문 구두점 포함)
    sent_lists = [re.split(r"[.!?…\n]+", t) for t in texts]
    for tok in tokens:
        tot = 0
        hit = 0
        pat = re.compile(re.escape(tok))
        for sents in sent_lists:
            for s in sents:
                ss = s.strip()
                if not ss:
                    continue
                if pat.search(ss):
                    tot += 1
                    if any(a in ss for a in anchors):
                        hit += 1
        rates[tok] = (hit / tot) if tot else 0.0
    return rates

def _mmr_select(candidates: list[str], cand_vecs: np.ndarray, query_vec: np.ndarray, k: int, lambda_div: float = 0.6) -> list[str]:
    if len(candidates) <= k:
        return candidates[:k]
    selected = []
    remain = list(range(len(candidates)))
    rel = (cand_vecs @ query_vec.T).ravel()
    first = int(np.argmax(rel))
    selected.append(first)
    remain.remove(first)
    while len(selected) < k and remain:
        sel_vecs = cand_vecs[selected]
        cand_vecs_r = cand_vecs[remain]
        max_red = (cand_vecs_r @ sel_vecs.T).max(axis=1)
        score = lambda_div*rel[remain] - (1-lambda_div)*max_red
        j = int(np.argmax(score))
        selected.append(remain[j])
        remain.pop(j)
    return [candidates[i] for i in selected]

class Embedder:
    def __init__(self):
        self.backend = None
        self.model = None
        self.is_e5 = False
        self.is_bge = False
        if cfg.EMB_BACKEND == "none":
            return
        backend = cfg.EMB_BACKEND
        if backend == "auto":
            backend = "bge" if HAS_BGE else ("e5" if HAS_ST else ("sbert" if HAS_ST else "none"))
        if backend == "bge" and HAS_BGE:
            name = cfg.EMB_MODEL or "BAAI/bge-m3"
            self.model = BGEM3FlagModel(name, use_fp16=True)
            self.backend = "bge"; self.is_bge = True
        elif backend in ("e5", "sbert") and HAS_ST:
            if backend == "e5":
                name = cfg.EMB_MODEL or "intfloat/multilingual-e5-base"
                self.is_e5 = True
            else:
                name = cfg.EMB_MODEL or "BM-K/KoSimCSE-roberta-multitask"
            self.model = SentenceTransformer(name)
            self.backend = backend
        else:
            self.backend = None

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        if self.model is None:
            return np.zeros((len(texts), 768), dtype=np.float32)
        if self.is_bge:
            out = self.model.encode(
                texts, batch_size=cfg.EMB_BATCH, max_length=512,
                return_dense=True, normalize_embeddings=True
            )
            v = out["dense_vecs"].astype(np.float32)
            return v
        to_enc = texts
        if self.is_e5:
            to_enc = [("query: " if is_query else "passage: ") + t for t in texts]
        v = self.model.encode(
            to_enc, batch_size=cfg.EMB_BATCH, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        return v

def variants_for(token: str) -> list[str]:
    t = token
    outs = {t}
    n = _norm_token(t)
    if n in {"ai","국가ai"} or t.lower() in {"ai","국가 ai","국가ai"}:
        outs |= {"AI","Ai","국가 ai","국가AI","국가 Ai"}
    if "차량용" in t or "차량 용" in t:
        outs |= {"차량용","차량 용"}
    if "레벨 4" in t or "레벨4" in t:
        outs |= {"레벨 4","레벨4","L4","Level 4","level 4"}
    if t.lower() == "ap":
        outs |= {"AP","오토파일럿","오토 파일럿"}
    return list(dict.fromkeys(outs))

def expand_with_variants(lst: list[str]) -> list[str]:
    bag = []
    for w in lst:
        bag += variants_for(w)
    return list(dict.fromkeys(bag))

def tokens_to_display_csv(tokens: list[str],
                          josa_pref: dict[str, dict] | None = None,
                          mode: str = "auto",
                          thresh: float = 0.60,
                          whitelist: set[str] | None = None,
                          show_bare_always: bool = True,
                          attach_josa: bool = True,
                          corpus_vocab: set[str] | None = None,
                          josa_min_count: int = 3,
                          dedup_stem: bool = True) -> str:
    shown = []
    seen_stems = set()
    wl = { _norm_token(w) for w in ((whitelist or set()) | PROPER_WHITELIST) }
    for t in tokens:
        base_raw = _norm_token(t)
        stem, _ = _split_josa(base_raw)
        if len(stem) <= 1 or stem in {"것","등","및"}:
            continue
        disp = _to_spaced_form(stem)
        best, ratio = "", 0.0
        if josa_pref and stem in josa_pref:
            meta = josa_pref[stem]
            per = meta.get("counts", {})
            total = float(meta.get("total", 1))
            cand = []
            for j, c in per.items():
                if c >= josa_min_count and (corpus_vocab is None or (stem + j) in corpus_vocab):
                    cand.append((c, j))
            cand.sort(reverse=True)
            if cand:
                best = cand[0][1]
                ratio = (per[best] / total) if total else 0.0
        is_proper = bool(PROPER_NOUN_HEUR.search(stem))
        if dedup_stem and stem in seen_stems:
            continue
        include = (mode == "always") or (mode == "auto" and ((stem in wl) or ((not is_proper) and ratio >= float(thresh))))
        if include or show_bare_always:
            shown.append(disp)
        if attach_josa and include and best:
            form = disp + best
            if (corpus_vocab is None) or (form in corpus_vocab):
                shown.append(form)
        if dedup_stem:
            seen_stems.add(stem)
    return ",".join(_dedup_preserve_order(shown))

def extract_keywords(df, min_df, top_n, ngram_max, neutral_band, ambig_unigrams=None):
    """자동차 도메인 편향 없이 키워드 추출 + 임베딩 재랭크 + MMR.
    내부 함수가 pos_top/neg_top을 참조하지 않도록 순서 및 스코프 재구성.
    """
    # --- 초기 가드 ---
    if "label" not in df.columns:
        return [], []
    pos_top, neg_top = None, None  # (중요) 사전 정의로 UnboundLocal 방지

    # 제목/리드 가중치
    title_w, lead_w = 3, 2
    titles = df["title"].fillna("").astype(str)
    leads = df["lead"].fillna("").astype(str) if "lead" in df.columns else pd.Series([""]*len(df))
    texts = ((titles + "\n")*title_w + (leads + "\n")*lead_w + df["content"].fillna("").astype(str)).apply(clean_text).tolist()
    y = df["label"].astype(int).to_numpy()

    vect = CountVectorizer(analyzer="word", token_pattern=TOKEN_PATTERNS["AN"],
                           ngram_range=(1, ngram_max), min_df=min_df)
    X = vect.fit_transform(texts)
    vocab = _vocab_from_vectorizer(vect)

    idx1, idx0 = (y == 1), (y == 0)
    if idx1.sum() == 0 or idx0.sum() == 0:
        return [], []

    # 통계량
    c1 = np.asarray(X[idx1].sum(axis=0)).ravel()
    c0 = np.asarray(X[idx0].sum(axis=0)).ravel()
    V = float(len(vocab))
    p1 = (c1 + 0.5) / (c1.sum() + 0.5 * V)
    p0 = (c0 + 0.5) / (c0.sum() + 0.5 * V)
    log_ratio = np.log(p1 / p0)
    lr_abs = np.abs(log_ratio)
    STRONG_THR = float(np.quantile(lr_abs, 0.70))

    neutral_set = set(vocab[np.abs(log_ratio) <= neutral_band])
    if ambig_unigrams is None:
        ambig_unigrams = set()

    idx_lut = {v: i for i, v in enumerate(vocab)}
    c_tot = c1 + c0

    # 컷 값(분포 기반)
    LR_TAU  = np.quantile(lr_abs, 0.70)    # 강한 단서 컷
    FREQ_TAU = np.quantile(c_tot, 0.60)    # 빈도 컷

    # ---- 안전한 허용 판별자들(절대 pos_top/neg_top 참조하지 않음) ----
    def _allow_token(t: str) -> bool:
        if t in neutral_set: return False
        if _is_noise_token(t): return False
        if _is_ambig_token(t): return False
        # 유니그램은 강단서/빈도 or 화이트리스트 필요
        if " " not in t and _norm_token(t) not in PROPER_WHITELIST:
            i = idx_lut.get(t, None)
            if i is None: return False
            if (lr_abs[i] < LR_TAU) and (c_tot[i] < FREQ_TAU):
                return False
        return True

    def _allow_idx(i: int) -> bool:
        t = vocab[i]
        n = _norm_token(t)
        unigram = (" " not in t)
        strong  = (lr_abs[i] >= STRONG_THR)

        # [ADD] 모호어(특히 AI 계열) 강력 차단 규칙
        if _is_ambig_token(t):
            # 1) 유니그램(단일 토큰)인 AI/인공지능은 무조건 불허
            if unigram:
                return False
            # 2) n-gram이어도 자동차 앵커와 같은 n-gram 안에 없으면 불허
            if not any(anc in n for anc in AUTO_ANCHORS_N):
                return False
            # 3) 여기까지 통과하면 허용(앵커 결합된 빅그램/트라이그램만)
            # 이후의 나머지 조건 체크 계속 진행

        if t in neutral_set:
            return False
        if _is_noise_token(t):
            return False
        if unigram and (n not in PROPER_WHITELIST) and (not strong):
            return False
        return True


    # 정렬 인덱스
    order_pos = np.argsort(-log_ratio)
    order_neg = np.argsort( log_ratio)

    # 1차 후보
    CAND = min(len(vocab), max(top_n * 30, 1800))
    cand_pos_idx = [i for i in order_pos[:CAND] if _allow_idx(i)]
    cand_neg_idx = [i for i in order_neg[:CAND] if _allow_idx(i)]

    # ---- 자동차 앵커 근접 소프트 게이트 ----
    cand_pos_tokens = [vocab[i] for i in cand_pos_idx]
    cand_neg_tokens = [vocab[i] for i in cand_neg_idx]

    cooc_pos_map = {}
    cooc_neg_map = {}
    if cand_pos_tokens:
        cooc_pos_map = _cooc_rate_bulk(texts, cand_pos_tokens, AUTO_ANCHORS)
    if cand_neg_tokens:
        cooc_neg_map = _cooc_rate_bulk(texts, cand_neg_tokens, AUTO_ANCHORS)

    def _dyn_thr(vals: list[float], base: float) -> float:
        arr = np.array(list(vals), dtype=float)
        if arr.size == 0: return base
        q = float(np.quantile(arr, 0.25))
        return max(base, q * 0.9)

    POS_THR = _dyn_thr(cooc_pos_map.values(), AUTO_POS_THR)
    NEG_THR = _dyn_thr(cooc_neg_map.values(), AUTO_NEG_THR)

    def _pass_auto_gate(token: str, i: int, mode: str) -> bool:
        co = (cooc_pos_map if mode=="pos" else cooc_neg_map).get(token, 0.0)
        n  = _norm_token(token)
        unigram = (" " not in token)
        strong  = (lr_abs[i] >= STRONG_THR)
        in_white= (n in { _norm_token(w) for w in PROPER_WHITELIST })
        is_ngram= (not unigram)
        thr = POS_THR if mode=="pos" else NEG_THR

        # [ADD] 모호어는 더 빡세게: (n-gram) AND (앵커 포함) AND (co-occur ≥ thr) 만 통과
        if _is_ambig_token(token):
            if not is_ngram:
                return False
            if not any(anc in n for anc in AUTO_ANCHORS_N):
                return False
            return co >= thr

        # 일반 토큰은 기존 완화 규칙 유지
        is_ambig = _is_ambig_token(token)
        return (co >= thr) or ((not is_ambig) and (strong or in_white or is_ngram))



    cand_pos_idx_soft = [i for i in cand_pos_idx if _pass_auto_gate(vocab[i], i, "pos")]
    cand_neg_idx_soft = [i for i in cand_neg_idx if _pass_auto_gate(vocab[i], i, "neg")]

    # 후보 과소 시 완화
    min_pos_cand = max(300, top_n * 2)
    min_neg_cand = max(300, top_n * 2)
    if len(cand_pos_idx_soft) < min_pos_cand: cand_pos_idx_soft = cand_pos_idx
    if len(cand_neg_idx_soft) < min_neg_cand: cand_neg_idx_soft = cand_neg_idx

    cand_pos_idx = cand_pos_idx_soft
    cand_neg_idx = cand_neg_idx_soft

    # ---- 임베딩 재랭킹 & MMR ----
    use_emb = bool(cfg.USE_EMB_RERANK)
    emb_w = float(getattr(cfg, "EMB_WEIGHT", 0.35))
    if use_emb:
        embedder = Embedder()
        if embedder.model is not None:
            doc_vecs = _l2_normalize(embedder.encode(texts, is_query=False))
            pos_cent = _l2_normalize(doc_vecs[idx1].mean(axis=0, keepdims=True))
            neg_cent = _l2_normalize(doc_vecs[idx0].mean(axis=0, keepdims=True))

            cand_pos_tokens = [vocab[i] for i in cand_pos_idx]
            cand_neg_tokens = [vocab[i] for i in cand_neg_idx]
            tok_pos_vecs = embedder.encode(cand_pos_tokens, is_query=True)
            tok_neg_vecs = embedder.encode(cand_neg_tokens, is_query=True)

            # 보조 co-occur(앵커)
            ANCHORS = ["자율주행","ADAS","SDV","레벨 4","완성차","현대차","기아","테슬라","EV","전기차","공장","양산","리콜"]
            cooc_pos_b = _cooc_rate_bulk(texts, cand_pos_tokens, ANCHORS)
            cooc_neg_b = _cooc_rate_bulk(texts, cand_neg_tokens, ANCHORS)
            W_C = 0.25

            def _norm(v):
                v = v.astype(np.float32)
                m = np.max(np.abs(v))
                return v / (m if m > 1e-8 else 1.0)

            lr_pos = _norm(log_ratio[cand_pos_idx])
            lr_neg = _norm(-log_ratio[cand_neg_idx])
            sim_pos_pos = _cos(tok_pos_vecs, pos_cent).ravel()
            sim_neg_pos = _cos(tok_pos_vecs, neg_cent).ravel()
            sim_pos_neg = _cos(tok_neg_vecs, pos_cent).ravel()
            sim_neg_neg = _cos(tok_neg_vecs, neg_cent).ravel()

            seed_key = f"{min_df}|{ngram_max}|{top_n}|{neutral_band:.6f}|{globals().get('RUN_SEED', cfg.RANDOM_STATE)}"
            jitter_pos = _tiny_hash_jitter(cand_pos_tokens, seed_key, scale=1e-3)
            jitter_neg = _tiny_hash_jitter(cand_neg_tokens, seed_key, scale=1e-3)

            score_pos = lr_pos + emb_w*(sim_pos_pos - sim_neg_pos) + W_C*np.array([cooc_pos_b[t] for t in cand_pos_tokens]) + jitter_pos
            score_neg = lr_neg + emb_w*(sim_neg_neg - sim_pos_neg) + W_C*np.array([1.0 - cooc_neg_b[t] for t in cand_neg_tokens]) + jitter_neg

            pos_sorted = np.argsort(-score_pos)
            neg_sorted = np.argsort(-score_neg)

            cand_pos_tokens = [cand_pos_tokens[i] for i in pos_sorted]
            cand_neg_tokens = [cand_neg_tokens[i] for i in neg_sorted]
            tok_pos_vecs = tok_pos_vecs[pos_sorted]
            tok_neg_vecs = tok_neg_vecs[neg_sorted]

            pos_top = _mmr_select(cand_pos_tokens, tok_pos_vecs, pos_cent, k=top_n, lambda_div=0.6)
            # 음성은 양성 충돌 제거 후 MMR
            pos_set = set(pos_top)
            mask_keep = [t not in pos_set for t in cand_neg_tokens]
            cand_neg_tokens2 = [t for t, keep in zip(cand_neg_tokens, mask_keep) if keep]
            tok_neg_vecs2 = tok_neg_vecs[mask_keep]
            neg_top = _mmr_select(cand_neg_tokens2, tok_neg_vecs2, neg_cent, k=top_n, lambda_div=0.6)

    # 해시 임베딩 폴백
    if pos_top is None or neg_top is None:
        cand_pos_tokens = [vocab[i] for i in cand_pos_idx]
        cand_neg_tokens = [vocab[i] for i in cand_neg_idx]
        tok_pos_vecs = _hash_token_vecs(cand_pos_tokens, dim=32)
        tok_neg_vecs = _hash_token_vecs(cand_neg_tokens, dim=32)
        w_pos = lr_abs[cand_pos_idx].astype(np.float32); w_pos /= (w_pos.max() + 1e-8)
        w_neg = lr_abs[cand_neg_idx].astype(np.float32); w_neg /= (w_neg.max() + 1e-8)
        pos_cent = _l2_normalize((tok_pos_vecs * w_pos[:, None]).sum(axis=0, keepdims=True))
        neg_cent = _l2_normalize((tok_neg_vecs * w_neg[:, None]).sum(axis=0, keepdims=True))
        pos_top = _mmr_select(cand_pos_tokens, tok_pos_vecs, pos_cent, k=top_n, lambda_div=0.6)
        # 음성은 양성 충돌 제거 후 MMR
        pos_set = set(pos_top)
        cand_neg_tokens2 = [t for t in cand_neg_tokens if t not in pos_set]
        tok_neg_vecs2 = _hash_token_vecs(cand_neg_tokens2, dim=32)
        neg_top = _mmr_select(cand_neg_tokens2, tok_neg_vecs2, neg_cent, k=top_n, lambda_div=0.6)

    # ----- 음성 폴백 강화: 너무 적으면 보충 -----
    if neg_top is None: neg_top = []
    need_neg_min = max(100, top_n // 2)
    if len(neg_top) < need_neg_min:
        # (중요) 여기서는 _allow_token만 쓰고, pos_top은 로컬 set로만 사용 → 스코프 안전
        base_neg = [vocab[i] for i in order_neg if _allow_token(vocab[i])]
        pos_set = set(pos_top or [])
        base_neg = [t for t in base_neg if t not in pos_set]
        if base_neg:
            Vn = _hash_token_vecs(base_neg, 32)
            qn = _l2_normalize(Vn.mean(axis=0, keepdims=True))
            neg_top = _mmr_select(base_neg, Vn, qn, k=min(top_n, len(base_neg)))

    # --- 반환형 보정(항상 list 보장) ---
    if pos_top is None: pos_top = []
    if neg_top is None: neg_top = []
    pos_top = [str(t) for t in pos_top]
    neg_top = [str(t) for t in neg_top]
    return pos_top, neg_top

def _cap_variants(tokens: list[str], per_token: int) -> list[str]:
    out, seen = [], {}
    for t in tokens:
        stem = _norm_token(_split_josa(t)[0])
        c = seen.get(stem, 0)
        if c < per_token:
            out.append(t); seen[stem] = c + 1
    return out

def _mmr_for_hints(tokens: list[str], k: int) -> list[str]:
    if not cfg.USE_HINT_MMR or len(tokens) <= k:
        return tokens[:k]
    rs = int(globals().get("RUN_SEED", cfg.RANDOM_STATE))
    tokens = sorted(tokens, key=lambda t: hashlib.md5((str(rs) + "|" + t).encode("utf-8")).hexdigest())
    try:
        emb = Embedder()
        if emb.model is not None:
            V = emb.encode(tokens, is_query=True)
            q = _l2_normalize(V.mean(axis=0, keepdims=True))
        else:
            V = _hash_token_vecs(tokens, 32)
            q = _l2_normalize(V.mean(axis=0, keepdims=True))
    except Exception:
        V = _hash_token_vecs(tokens, 32)
        q = _l2_normalize(V.mean(axis=0, keepdims=True))
    return _mmr_select(tokens, V, q, k=k, lambda_div=0.6)

def _mix_force_and_learned(learned: list[str], forced: list[str], k: int, force_budget: int) -> list[str]:
    base_force = list(dict.fromkeys(forced))[:min(force_budget, k)]
    rem_k = max(0, k - len(base_force))
    learned_clean = [t for t in learned if _norm_token(t) not in { _norm_token(x) for x in base_force }]
    picked = _mmr_for_hints(learned_clean, rem_k)
    return base_force + picked

def assemble_prompt(pos_kw: list[str], neg_kw: list[str], max_chars: int, hint_k: int, josa_pref: dict):
    global corpus_vocab
    core = COMPACT_CORE_TMPL
    parts = [core]

    # 0) 최종 k
    k = min(max(35, int(hint_k)), int(getattr(cfg, "HINT_BUDGET", 65)))
    k_pos = k
    k_neg = max(k, int(getattr(cfg, "MIN_NEG_HINTS", 20)))  # 제외 최소치 보장

    # 포함/제외 강제 셋의 교차 오염 방지(토큰 추가 없이 필터만)
    neg_force_stems = { _norm_token(_split_josa(x)[0]) for x in FORCE_NEG_EXTRA }
    pos_force_stems = { _norm_token(_split_josa(x)[0]) for x in FORCE_POS_EXTRA }
    pos_kw = [t for t in pos_kw if _norm_token(_split_josa(t)[0]) not in neg_force_stems]
    neg_kw = [t for t in neg_kw if _norm_token(_split_josa(t)[0]) not in pos_force_stems]

    pos_kw = [t for t in pos_kw if not _is_noise_token(t)]
    neg_kw = [t for t in neg_kw if not _is_noise_token(t)]

    # 강제 예산
    force_pos_budget = min(cfg.FORCE_BUDGET_POS, max(1, int(k * 0.10)))
    force_neg_budget = max(cfg.FORCE_BUDGET_NEG, max(1, int(k * cfg.NEG_FORCE_RATIO)))

    # 1) 모호어 제거
    pos_kw = _filter_moho(pos_kw)
    neg_kw = _filter_moho(neg_kw)

    # 2) 강제 + 학습 결합
    pos_sel = _mix_force_and_learned(pos_kw, FORCE_POS_EXTRA, k_pos, force_pos_budget)
    neg_sel = _mix_force_and_learned(neg_kw, FORCE_NEG_EXTRA, k_neg, force_neg_budget)

    # '학습된' 제외 힌트 최소 쿼터 보장
    neg_learn_min = max(1, int(k_neg * cfg.NEG_LEARNED_MIN_RATIO))
    neg_force_set = { _norm_token(x) for x in FORCE_NEG_EXTRA }
    neg_learned_now = [t for t in neg_sel if _norm_token(t) not in neg_force_set]
    if len(neg_learned_now) < neg_learn_min:
        pool = [t for t in neg_kw if _norm_token(t) not in { _norm_token(x) for x in neg_sel }]
        topup = _mmr_for_hints(pool, neg_learn_min - len(neg_learned_now))
        neg_sel = _dedup_preserve_order(neg_sel + topup)

    # 3) 변이 정책
    if cfg.VARIANT_POLICY == "after":
        pos_show = expand_with_variants(pos_sel)
        neg_show = expand_with_variants(neg_sel)
    elif cfg.VARIANT_POLICY == "learned_only":
        pos_show = expand_with_variants(pos_sel[:-force_pos_budget]) + pos_sel[-force_pos_budget:]
        neg_show = expand_with_variants(neg_sel[:-force_neg_budget]) + neg_sel[-force_neg_budget:]
    else:
        pos_show, neg_show = pos_sel, neg_sel

    if cfg.VARIANT_POLICY != "none":
        pos_show = _cap_variants(pos_show, cfg.VARIANT_PER_TOKEN)
        neg_show = _cap_variants(neg_show, cfg.VARIANT_PER_TOKEN)

    # 다양성 선택 (여기서 목표 개수 적용)
    pos_show = _mmr_for_hints(pos_show, k_pos)
    neg_show = _mmr_for_hints(neg_show, k_neg)

    # # (중요) 양성 힌트에서 모호어 강제 제거 (이중 안전장치)
    pos_show = _dedup_by_alias(pos_show)
    neg_show = _dedup_by_alias(neg_show)

    neg_show = [ t for t in neg_show if _alias_key(t) != "ai"]

    ANCHOR_SET = { _norm_token(w) for w in PROPER_WHITELIST }
    pos_keys = [_alias_key(t) for t in pos_show]
    neg_keys = [_alias_key(t) for t in neg_show]
    conf_keys = set(pos_keys) & set(neg_keys)

    if conf_keys:
        keep_pos, drop_neg_keys = [], set()
        for t in pos_show:
            k = _alias_key(t)
            if k in conf_keys:
                # 앵커(브랜드/차명 등) 또는 n-gram(공백 포함)은 '포함' 우선
                if (_norm_token(_split_josa(t)[0]) in ANCHOR_SET) or (" " in t):
                    keep_pos.append(t)
                    drop_neg_keys.add(k)
                # 그 외는 '제외' 우선 → 포함에서 제거
            else:
                keep_pos.append(t)
        pos_show = keep_pos
        neg_show = [t for t in neg_show if _alias_key(t) not in drop_neg_keys]


    # k 보장(양성)
    if len(pos_show) < k_pos:
        used = { _alias_key(x) for x in pos_show }
        pool = [t for t in pos_kw if _norm_token(t) not in used]
        pos_show = pos_show + _mmr_for_hints(pool, k_pos - len(pos_show))

    # k 보장(음성: 최소치까지)
    if len(neg_show) < k_neg:
        used = { _alias_key(x) for x in neg_show }
        pool = [t for t in neg_kw if _norm_token(t) not in used]
        neg_show = neg_show + _mmr_for_hints(pool, k_neg - len(neg_show))

    pos_show = [t for t in pos_show if _alias_key(t) not in {"ai"}]

    # ▼ CSV 게이트 완화: 포함은 bare 허용(thresh ↓), 제외는 'always'+bare 허용
    pk = tokens_to_display_csv(
        pos_show, josa_pref=josa_pref, mode="auto", whitelist=PROPER_WHITELIST,
        corpus_vocab=corpus_vocab if 'corpus_vocab' in globals() else None,
        josa_min_count=3, show_bare_always=True, thresh=0.35
    )
    nk = tokens_to_display_csv(
        neg_show, josa_pref=josa_pref, mode="always", whitelist=PROPER_WHITELIST,
        corpus_vocab=corpus_vocab if 'corpus_vocab' in globals() else None,
        josa_min_count=0, show_bare_always=True
    )

    parts.append(f"포함:{pk}\n제외:{nk}")
    return "\n\n".join(parts)[:max_chars]

def make_groups(df: pd.DataFrame) -> np.ndarray:
    def _normalize(s: str): return re.sub(r"\s+", " ", (s or "").lower()).strip()
    base = (df["title"].fillna("").astype(str).apply(_normalize) + "||" + df["content"].fillna("").astype(str).apply(_normalize))
    return np.array([int(hashlib.md5(x.encode("utf-8")).hexdigest(), 16) % (10**9) for x in base])

def vectorize_train(train_texts, token_pat, ngram_max, min_df):
    vect = CountVectorizer(analyzer="word", token_pattern=token_pat, ngram_range=(1, ngram_max), min_df=min_df, binary=True)
    return vect, vect.fit_transform(train_texts)

def kw_train_weights(Xtr, ytr):
    ytr = np.asarray(ytr, dtype=int)
    idx1, idx0 = (ytr==1), (ytr==0)
    c1 = np.asarray(Xtr[idx1].sum(axis=0)).ravel().astype(float)
    c0 = np.asarray(Xtr[idx0].sum(axis=0)).ravel().astype(float)
    V = float(Xtr.shape[1]); alpha = 0.5
    p1 = (c1 + alpha) / (c1.sum() + alpha*V); p0 = (c0 + alpha) / (c0.sum() + alpha*V)
    w = np.log(p1/p0); tau_prior = np.log((c0.sum()+alpha*V)/(c1.sum()+alpha*V))
    return w, tau_prior

def kw_predict(vect, w, tau, texts):
    scores = vect.transform(texts) @ w
    yhat = (scores - tau >= 0).astype(int)
    return np.asarray(yhat).ravel(), np.asarray(scores).ravel()

def best_tau_for_accuracy(scores: np.ndarray, y_true: np.ndarray) -> float:
    s, y = np.asarray(scores), np.asarray(y_true)
    if s.size == 0: return 0.0
    uniq = np.unique(s)
    if uniq.size <= 1: return (uniq[0] - 1.0) if uniq.size > 0 else 0.0
    candidates = ((uniq[:-1] + uniq[1:]) * 0.5).tolist()
    best_acc, best_tau = -1.0, uniq[0]
    for t in candidates:
        acc = ((s >= t).astype(int) == y).mean()
        if acc > best_acc: best_acc, best_tau = acc, t
    return best_tau

def _simhash64(s: str) -> int:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", s.lower())
    v = np.zeros(64, dtype=int)
    for tok in set(tokens):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) & ((1<<64)-1)
        for b in range(64):
            v[b] += 1 if (h>>b)&1 else -1
    out = 0
    for b in range(64):
        if v[b] >= 0: out |= (1<<b)
    return out

def _dedup_by_title_simhash(df: pd.DataFrame, ham_thres: int = 3) -> pd.DataFrame:
    if "title" not in df.columns:
        return df
    hashes = df["title"].fillna("").astype(str).apply(_simhash64).tolist()
    keep, seen = [], []
    for i, h in enumerate(hashes):
        if any(bin(h ^ hh).count("1") <= ham_thres for hh in seen):
            continue
        seen.append(h); keep.append(i)
    return df.iloc[keep].reset_index(drop=True)

def build_corpus_vocab(df, positive_only=True) -> set[str]:
    if positive_only and 'label' in df.columns:
        df = df[df['label'] == 1]
    texts = (df["title"].fillna("") + "\n" + df["content"].fillna("")).apply(clean_text).tolist()
    vect = CountVectorizer(analyzer="word", token_pattern=TOKEN_PATTERNS["AN"])
    _ = vect.fit_transform(texts)
    return set(_vocab_from_vectorizer(vect))

def build_josa_preference(df, all_keywords):
    josa_pref = {}
    normalized_keywords = { _norm_token(w) for w in all_keywords }
    df_j = df[df['label'] == 1]
    texts_j = (df_j["title"].fillna("") + "\n" + df_j["content"].fillna("")).apply(clean_text)
    vect = CountVectorizer(analyzer="word", token_pattern=TOKEN_PATTERNS["AN"])
    X = vect.fit_transform(texts_j)
    vocab = _vocab_from_vectorizer(vect)
    counts = dict(zip(vocab, np.asarray(X.sum(axis=0)).ravel()))
    stem_counts = Counter()
    josa_counts = Counter()
    for token, count in counts.items():
        stem, josa = _split_josa(token)
        if stem in normalized_keywords:
            stem_counts[stem] += count
            if josa:
                josa_counts[(stem, josa)] += count
    for stem, total in stem_counts.items():
        if total <= 0:
            continue
        best_josa, best_count = "", 0
        per_josa = {}
        for (s, j), c in josa_counts.items():
            if s == stem:
                per_josa[j] = c
                if c > best_count:
                    best_count, best_josa = c, j
        ratio = (best_count / total) if total else 0.0
        josa_pref[stem] = {
            "best": best_josa, "ratio": ratio, "total": total, "counts": per_josa,
        }
    return josa_pref

def length_score_official(chars: int) -> float:
    L = min(chars, 3000)
    return (1 - (L / 3000)**2)**0.5

def build_fixed_splits(y: np.ndarray,
                       groups: np.ndarray,
                       n_splits: int,
                       seed: int,
                       path: str):
    """
    고정 스플릿을 만들어 캐시에 저장/로드.
    - 신 포맷(dict): {"_ver":1, "_sig":{...}, "splits":[{"tr":[...],"va":[...]}, ...]}
    - 구 포맷(list): [{"tr":[...],"va":[...]}, ...]  ← 자동 마이그레이션
    """
    p = Path(path)
    sig = {
        "n": int(len(y)),
        "sumy": int(np.asarray(y, dtype=int).sum()),
        "ns": int(n_splits),
        "seed": int(seed),
        "arr": _arr_sig(y, groups)
    }

    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                splits_legacy = []
                for d in data:
                    if isinstance(d, dict) and "tr" in d and "va" in d:
                        tr = np.array(d["tr"], dtype=int)
                        va = np.array(d["va"], dtype=int)
                        splits_legacy.append((tr, va))
                to_save = {
                    "_ver": 1,
                    "_sig": sig,
                    "splits": [{"tr": tr.tolist(), "va": va.tolist()} for tr, va in splits_legacy],
                }
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(to_save, ensure_ascii=False), encoding="utf-8")
                return splits_legacy

            if isinstance(data, dict):
                dsig = data.get("_sig", {})
                if (
                    dsig.get("n") == sig["n"]
                    and dsig.get("sumy") == sig["sumy"]
                    and dsig.get("ns") == sig["ns"]
                    and dsig.get("seed") == sig["seed"]
                    and (
                        # 과거 캐시에 arr가 없을 수도 있으니, 있으면 비교 / 없으면 통과
                        ("arr" not in dsig) or (dsig.get("arr") == sig["arr"])
                    )
                ):
                    items = data.get("splits", [])
                    splits = []
                    for item in items:
                        tr = np.array(item["tr"], dtype=int)
                        va = np.array(item["va"], dtype=int)
                        splits.append((tr, va))
                    return splits
                else:
                    print("[INFO] split cache signature mismatch → regenerate.")
        except Exception as e:
            print(f"[WARN] split cache unreadable ({e}) → regenerate.")

    if HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = splitter.split(np.zeros(len(y)), y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        it = splitter.split(np.zeros(len(y)), groups=groups)

    splits = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in it]

    to_save = {
        "_ver": 1,
        "_sig": sig,
        "splits": [{"tr": tr.tolist(), "va": va.tolist()} for tr, va in splits],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_save, ensure_ascii=False), encoding="utf-8")

    return splits

def ray_prime_candidates(df, texts_all, y_all, splits, josa_pref):
    if not cfg.USE_RAY_PRIME:
        return []
    try:
        import ray
        from ray import tune
        if cfg.RAY_SCHEDULER.upper() == "BOHB":
            from ray.tune.schedulers import HyperBandForBOHB
            scheduler = HyperBandForBOHB()
        else:
            from ray.tune.schedulers import ASHAScheduler
            scheduler = ASHAScheduler(max_t=len(splits), grace_period=1, reduction_factor=2)
        ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")

        def _tune_obj(config):
            top_n = int(config["TOP_N"])
            min_df = int(config["MIN_DF"])
            ngram_max = int(config["NGRAM_MAX"])
            neutral_band = float(config["NEUTRAL_BAND"])
            hint_k = int(config["HINT_K"])

            pos_kw, neg_kw = extract_keywords(df, min_df, top_n, ngram_max, neutral_band)
            prompt_text = assemble_prompt(pos_kw, neg_kw, cfg.CHAR_LIMIT, hint_k, josa_pref)
            len_score = length_score_official(len(prompt_text))

            acc_sum = 0.0
            max_steps = max(1, len(splits)//2)
            for step, (tr, va) in enumerate(splits[:max_steps], start=1):
                tr_texts = [texts_all[i] for i in tr]
                va_texts = [texts_all[i] for i in va]
                y_tr, y_va = y_all[tr], y_all[va]
                vect, Xtr = vectorize_train(tr_texts, TOKEN_PATTERNS["AN"], ngram_max, max(1, min_df))
                w, _ = kw_train_weights(Xtr, y_tr)
                _, scores_va = kw_predict(vect, w, 0.0, va_texts)
                tau = best_tau_for_accuracy(scores_va, y_va)
                yhat_va = (scores_va - tau >= 0).astype(int)
                fold_acc = (yhat_va == y_va).mean()
                acc_sum += fold_acc
                cur_acc = acc_sum / step
                cur_score = 0.9 * cur_acc + 0.1 * len_score
                tune.report(score=cur_score, step=step)

        # Ray search space도 Optuna와 완전 일치
        search_space = {
            "TOP_N": tune.randint(200, 501),        # hi는 exclusive일 수 있어 +1
            "MIN_DF": tune.randint(2, 7),
            "NGRAM_MAX": tune.randint(2, 6),
            "NEUTRAL_BAND": tune.uniform(0.07, 0.40),
            "HINT_K": tune.randint(45, 61),
        }

        analysis = tune.run(
            _tune_obj,
            config=search_space,
            metric="score",
            mode="max",
            num_samples=cfg.RAY_PRIME_NUM_SAMPLES,
            time_budget_s=cfg.RAY_PRIME_TIME_S,
            scheduler=scheduler,
            verbose=0,
        )
        dfres = analysis.results_df.sort_values("score", ascending=False).head(cfg.RAY_PRIME_TOPK)
        cand = []
        for _, r in dfres.iterrows():
            cand.append({
                "TOP_N": int(r["config/TOP_N"]),
                "MIN_DF": int(r["config/MIN_DF"]),
                "NGRAM_MAX": int(r["config/NGRAM_MAX"]),
                "NEUTRAL_BAND": float(r["config/NEUTRAL_BAND"]),
                "HINT_K": int(r["config/HINT_K"]),
            })
        try: ray.shutdown()
        except: pass
        return cand
    except Exception as e:
        print(f"[WARN] Ray Tune 프리패스 건너뜀: {e}")
        return []

# ---------- 3) Objective ----------
def _group_macro_acc(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray) -> float:
    vals = []
    for g in np.unique(group_ids):
        m = (group_ids == g)
        if m.sum() == 0:
            continue
        vals.append((y_pred[m] == y_true[m]).mean())
    return float(np.mean(vals)) if vals else 0.0

def objective(trial: optuna.Trial,
              df,
              texts_all,
              y_all,
              groups,
              josa_pref,
              splits) -> float:
    # ---- 하이퍼 파라미터 샘플링(고정 정의 경유) ----
    top_n        = _suggest_from_space(trial, "TOP_N")
    min_df       = _suggest_from_space(trial, "MIN_DF")
    ngram_max    = _suggest_from_space(trial, "NGRAM_MAX")
    neutral_band = _suggest_from_space(trial, "NEUTRAL_BAND")
    hint_k       = _suggest_from_space(trial, "HINT_K")

    # ---- 키워드 추출 & 프롬프트 조립 ----
    pos_kw, neg_kw = extract_keywords(df, min_df, top_n, ngram_max, neutral_band)
    prompt_text = assemble_prompt(pos_kw, neg_kw, cfg.CHAR_LIMIT, hint_k, josa_pref)
    len_score = length_score_official(len(prompt_text))

    # ---- 고정 스플릿으로 OOF 산출 (groups 명시적 사용) ----
    oof_pred = np.zeros_like(y_all, dtype=int)
    fold_scores = []

    for step, (tr, va) in enumerate(splits):
        if len(np.intersect1d(groups[tr], groups[va])) > 0:
            raise TrialPruned("Group leakage detected in provided splits.")

        tr_texts = [texts_all[i] for i in tr]
        va_texts = [texts_all[i] for i in va]
        y_tr, y_va = y_all[tr], y_all[va]
        g_va = groups[va]

        vect, Xtr = vectorize_train(tr_texts, TOKEN_PATTERNS["AN"], ngram_max, max(1, min_df))
        w, _ = kw_train_weights(Xtr, y_tr)
        _, scores_va = kw_predict(vect, w, 0.0, va_texts)
        tau = best_tau_for_accuracy(scores_va, y_va)

        yhat_va = (scores_va - tau >= 0).astype(int)
        oof_pred[va] = yhat_va

        fold_acc = _group_macro_acc(y_va, yhat_va, g_va)
        fold_scores.append(fold_acc)

        trial.report(0.9 * fold_acc + 0.1 * len_score, step=step + 1)
        if trial.should_prune():
            raise TrialPruned()

    acc_macro = float(np.mean(fold_scores)) if fold_scores else 0.0
    final_score = 0.9 * acc_macro + 0.1 * len_score
    return final_score

# ---------- 4) Main ----------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = Config()
    cfg.USE_RAY_PRIME = True

    run_seed = int(datetime.datetime.now().timestamp()) % 10_000_000
    RUN_SEED = run_seed
    globals()["RUN_SEED"] = run_seed
    np.random.seed(run_seed)
    random.seed(run_seed)
    print(f"[seed] run_seed={run_seed}")

    base_path = Path(cfg.PATH).resolve()
    csv_path = base_path / cfg.CSV_NAME
    if not csv_path.exists():
        print(f"[ERR] CSV 파일을 찾지 못했습니다: {csv_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"[OK] 데이터 로드 완료: {csv_path} ({len(df)} rows)")

    df = _dedup_by_title_simhash(df, ham_thres=3)
    print(f"[OK] 중복 제거 후: {len(df)} rows")

    build_variant_pairs_from_data(df)
    corpus_vocab = build_corpus_vocab(df, positive_only=True)

    texts_all = (df["title"].fillna("") + "\n" + df["content"].fillna("")).apply(clean_text).tolist()
    y_all = df["label"].astype(int).to_numpy()
    groups = make_groups(df)

    splits = build_fixed_splits(y_all, groups, cfg.N_SPLITS, cfg.RANDOM_STATE, cfg.SPLIT_CACHE)
    print(f"[splits] {len(splits)} folds prepared.")

    temp_pos_kw, temp_neg_kw = extract_keywords(df, 2, 500, 2, 0.25)
    
    temp_pos_kw = temp_pos_kw or []
    temp_neg_kw = temp_neg_kw or []
    all_kw_for_josa = temp_pos_kw + temp_neg_kw
    
    josa_pref = build_josa_preference(df, temp_pos_kw + temp_neg_kw)

    # ---- Sampler (버전 호환: 지원되면 warn_independent_sampling 끄고, 안 되면 생략) ----
    sampler_kwargs = dict(
        seed=run_seed,
        n_startup_trials=cfg.N_STARTUP,
        multivariate=cfg.MULTIVARIATE,
        group=cfg.MULTIVARIATE,
    )
    try:
        sampler = optuna.samplers.TPESampler(**sampler_kwargs, constant_liar=cfg.CONSTANT_LIAR, warn_independent_sampling=False)
    except TypeError:
        # group/constant_liar/warn_independent_sampling 중 미지원인 경우 단계적으로 제거
        sampler_kwargs.pop("group", None)
        try:
            sampler = optuna.samplers.TPESampler(**sampler_kwargs, warn_independent_sampling=False)
        except TypeError:
            try:
                sampler = optuna.samplers.TPESampler(**sampler_kwargs)
            except TypeError:
                sampler = optuna.samplers.TPESampler(seed=run_seed)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(1, len(splits)//2)) if cfg.USE_PRUNER else None

    # ---- Study 이름에 서치스페이스 시그니처 부여(과거 DB와 절연) ----
    study_name = f"{cfg.STUDY_NAME}_{SPACE_SIG}"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=cfg.STORAGE,
        study_name=study_name,
        load_if_exists=True,
    )

    # ---- 서치스페이스 헬스체크(옵션) ----
    try:
        from optuna.search_space import intersection_search_space
        def _log_space_health(study):
            rss = intersection_search_space(study)
            miss = sorted(set(SEARCH_SPACE.keys()) - set(rss.keys()))
            if miss:
                print(f"[WARN] params falling back to independent sampling: {miss}")
            else:
                print("[OK] all params modeled jointly by multivariate TPE.")
        print("[space] before optimize:"); _log_space_health(study)
    except Exception:
        pass

    # ---- Ray 프리패스 → 검색공간 일치 검증 후 enqueue ----
    if cfg.USE_RAY_PRIME:
        prime_params = ray_prime_candidates(df, texts_all, y_all, splits, josa_pref)
        cleaned = []
        for p in prime_params:
            q = _coerce_to_search_space(p)
            if q is not None:
                cleaned.append(q)
        print(f"[ray] prime enqueue (clean) = {len(cleaned)}/{len(prime_params)}")
        for q in cleaned:
            study.enqueue_trial(q)

    # ---- 웜스타트도 동일 검증 ----
    try:
        ws = Path(cfg.WARM_START_FILE)
        if ws.exists():
            best_prev = json.loads(ws.read_text(encoding="utf-8"))
            best_prev = _coerce_to_search_space(best_prev)
            if best_prev is not None:
                study.enqueue_trial(best_prev)
                print(f"[OK] Warm-start enqueued: {best_prev}")
            else:
                print("[WARN] Warm-start skipped (out-of-space)")
    except Exception as e:
        print(f"[WARN] Warm-start load fail: {e}")

    objective_with_data = lambda trial: objective(trial, df, texts_all, y_all, groups, josa_pref, splits)
    print(f"\n>> Optuna 최적화 시작 (trials={cfg.OPTUNA_TRIALS}) ...")
    study.optimize(objective_with_data, n_trials=cfg.OPTUNA_TRIALS, show_progress_bar=True)

    try:
        print("[space] after optimize:"); _log_space_health(study)  # if available
    except Exception:
        pass

    Path(cfg.WARM_START_FILE).write_text(
        json.dumps(study.best_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Warm-start saved: {cfg.WARM_START_FILE}")

    print("\n" + "="*40)
    print("      최적화 완료!       ")
    print("="*40)
    print(f"최고 점수: {study.best_value:.4f}")
    print("최적의 하이퍼파라미터:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  - {key}: {value}")

    # 최적 파라미터로 프롬프트 생성/저장
    pos_kw, neg_kw = extract_keywords(
        df, best_params['MIN_DF'], best_params['TOP_N'],
        best_params['NGRAM_MAX'], best_params['NEUTRAL_BAND']
    )
    josa_pref_final = build_josa_preference(df, pos_kw + neg_kw)

    hint_k_fixed = max(best_params.get('HINT_K', 45), 45)
    prompt_text = assemble_prompt(
        pos_kw, neg_kw, cfg.CHAR_LIMIT, hint_k_fixed, josa_pref_final
    )

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_filename = f"dacon_system_prompt_OPTIMIZED_{study.best_value:.4f}_{now}.txt"
    out_path = Path(cfg.SAVE_PATH).resolve() / out_filename
    out_path.write_text(prompt_text, encoding="utf-8")
    print(f"[OK] 최종 시스템 프롬프트 저장 완료: {out_path}")

    print(f"\n--- 최종 산정 방식 ---")
    print(f"1. Optuna로 프롬프트 생성 규칙 자동 탐색 + 프루닝")
    print(f"2. 키워드 자동 추출(log-odds + 임베딩 재랭킹 + MMR)")
    print(f"3. 고정 스플릿 CV로 점수 산정, 폴드별 report/prune")
    print(f"4. 최종 점수 = 0.9*Accuracy + 0.1*LengthScore")
