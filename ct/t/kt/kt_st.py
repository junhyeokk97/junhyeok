#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
1) samples.csv(user_prompt, output) → 라벨셋/단서 자동 추출 → "강화된" 시스템 프롬프트 생성/저장
2) 규칙(통계)기반 다속성 분류(기본) + (옵션) 부스팅(LightGBM) 모델
3) K-Fold OOF 평가 스코어 → system_prompt / submission 파일명에 SCORE+TIMESTAMP 반영
"""

from __future__ import annotations

# =========================
# Imports
# =========================
from pathlib import Path
import sys, os, re, warnings
from typing import List, Tuple, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import html
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# =========================
# Config
# =========================
class Config:
    PATH = './ct/t/kt'
    CSV_NAME = 'samples.csv'
    SAVE_PATH = './ct/t/kt'

    # 프롬프트
    PROMPT_MAX_CHARS = 1200
    EXAMPLES_PER_PROMPT = 5

    # 단서/프롬프트용 토큰화
    # Config 안
    TOKEN_PATTERN = r"(?u)(?:<[^>]+>|[가-힣]{2,}|[A-Za-z]{2,}|[0-9]+(?:\.[0-9]+)?)"

    STOP_WORDS = []
    MIN_DF = 2
    MAX_DF = 1.0
    MIN_N = 1
    MAX_N = 3         # ← n-gram 3까지 확장
    MAX_FEATURES = None

    # 부스팅
    USE_BOOSTING = True
    BOOSTING_PARAMS = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.6,
        "bagging_freq": 3,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 193
    }

    # 평가
    N_SPLITS = 4
    USE_RULE_OVERRIDES = True

cfg = Config()
os.makedirs(cfg.SAVE_PATH, exist_ok=True)

# =========================
# Normalization (정규화 매핑)
# =========================
# 의미범주 토큰으로 묶어 중복/변형을 정리
NORM_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 공백/기호
    (re.compile(r"\s+"), " "),
    (re.compile(r"[“”\"‟＂]"), '"'),
    (re.compile(r"[’‘＇]"), "'"),
    (re.compile(r"[‐–—]"), "-"),
    # 흔한 오인식
    (re.compile(r"([가-힣]+)할 있다"), r"\1할 수 있다"),
    (re.compile(r"([가-힣]+)길 있다"), r"\1길 수 있다"),
    # 의미 카테고리
    (re.compile(r"(할\s*예정|할\s*계획|될\s*전망|될\s*것|되겠다|있겠다|예상|전망|목표|추진할\s*계획|선보일\s*예정|출시할\s*예정)"), "<FUTURE_PLAN>"),
    (re.compile(r"(출시한다|선보인다|개최한다|오픈한다|진행한다)"), "<SCHEDULE_NOW>"),
    (re.compile(r"(발표했다|밝혔다|전했다|기록했다|했었|했다|였다|되었다|이었다|였다)"), "<PAST_REPORT>"),
    (re.compile(r"(\?|왜|무엇|어떻게|얼마나|몇\s*시|누가|어디|인가요|습니까\?)"), "<INTERACT>"),
    (re.compile(r"(십시오|하세요|합시다|하라|요청|권고|주의|안내)"), "<INTERACT>"),
    (re.compile(r"(아니다|않|못|없|부진|감소|하락|불가|불가능)"), "<NEG>"),
    (re.compile(r"(수\s*있다|가능성|전망|예상|의도|계획|추정|관측|것\s*같다|보인다|듯하다)"), "<UNCERT>"),
    # 시간 표지 (추가)
    (re.compile(r"(내일|모레|내년|내달|다음\s*달|다음\s*주|오는|다가오는|향후|연말|하반기|상반기)"), "<TIME_FUT>"),
    (re.compile(r"(어제|그제|지난\s*달|지난\s*주|작년|전날|방금|조금\s*전)"), "<TIME_PAST>"),
    (re.compile(r"(현재|지금|요즘|~중|중이다|하고\s*있다|진행\s*중)"), "<TIME_NOW>"),

    # 미래 의도/계획/전망 → FUTURE (유지+보강)
    (re.compile(r"(할\s*예정|할\s*계획|될\s*전망|될\s*것|되겠다|있겠다|예상|전망|목표|의도|계획|추진할\s*계획|선보일\s*예정|출시할\s*예정)"),
     "<FUTURE_PLAN>"),

    # 일정성 현재형(보강)
    (re.compile(r"(출시한다|선보인다|개최한다|오픈한다|진행한다)"), "<SCHEDULE_NOW>"),

    # 과거 보고(유지)
    (re.compile(r"(발표했다|밝혔다|전했다|기록했다|했었|했다|였다|되었다|이었다)"), "<PAST_REPORT>"),

    # 상호작용(유지)
    (re.compile(r"(\?|왜|무엇|어떻게|얼마나|몇\s*시|누가|어디|인가요|습니까\?|십시오|하세요|합시다|하라|요청|권고|주의|안내)"),
     "<INTERACT>"),

    # 부정(유지)
    (re.compile(r"(아니다|않|못|없|부진|감소|하락|불가|불가능)"), "<NEG>"),

    # 불확실(계획/의도/전망 제거 → HEDGE로 분리)
    (re.compile(r"(수\s*있다|가능성|추정|관측|것\s*같다|보인다|듯하다)"), "<HEDGE>"),
]

def normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F\u200B-\u200F\uFEFF]", "", s)
    for pat, rep in NORM_PATTERNS:
        s = pat.sub(rep, s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# 규칙 평가용 정규표현(원문/정규화 토큰 둘 다 커버)
TEN_PAST   = re.compile(r"(?:<PAST_REPORT>|<TIME_PAST>)")
TEN_FUTURE = re.compile(r"(?:<FUTURE_PLAN>|<TIME_FUT>)")
TEN_NOW    = re.compile(r"(?:<TIME_NOW>)")
TYPE_INTERACT = re.compile(r"(?:<INTERACT>)")
TYPE_PREDICT  = re.compile(r"(?:<FUTURE_PLAN>)")
TYPE_INFER    = re.compile(r"(?:<HEDGE>)")
POL_NEG       = re.compile(r"(?:<NEG>)")
UNCERTAIN     = re.compile(r"(?:<HEDGE>)")   # ← 여기서 FUTURE_PLAN 제외!
TEN_SCHED  = re.compile(r"(?:<SCHEDULE_NOW>)")

# ===== (NEW) Token-based tense helper =====
def _tense_label_from_norm(norm: str) -> str:
    """정규화 텍스트(norm)에 포함된 의미 토큰으로 시제 결정(우선순위: 과거 > 미래 > 현재)."""
    has_past = bool(TEN_PAST.search(norm))
    has_future = bool(TEN_FUTURE.search(norm) or TEN_SCHED.search(norm))
    if has_past and has_future:
        # 보도+전망 같이 있을 때 과거 우선 (ex. '전망을 발표했다' → 과거)
        return "과거"
    if has_past:
        return "과거"
    if has_future:
        return "미래"
    return "현재"

# =========================
# Data
# =========================
def load_data(cfg: Config) -> pd.DataFrame:
    """CSV -> DataFrame(user_prompt, output → type, polarity, tense, certainty)"""
    df = pd.read_csv(Path(cfg.PATH) / cfg.CSV_NAME)
    # (선택) 특정 행 제외가 필요하면 여기에 인덱스 지정
    # rows_to_exclude = [10, 18, 19]
    # df = df.drop(index=[i for i in rows_to_exclude if i in df.index]).copy()

    df['user_prompt'] = df['user_prompt'].astype(str).map(normalize_text)
    df['output'] = df['output'].astype(str).str.strip('"')
    parts = df['output'].str.split(',', expand=True)
    if parts.shape[1] != 4:
        print("[ERR] output 형식은 '유형,극성,시제,확실성' 4필드여야 합니다.", file=sys.stderr); sys.exit(2)
    parts.columns = ['type','polarity','tense','certainty']
    for c in parts.columns:
        df[c] = parts[c].astype(str).str.strip()
    df = df.reset_index(drop=True)
    return df

def build_label_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        'type': sorted(df['type'].unique().tolist()),
        'polarity': sorted(df['polarity'].unique().tolist()),
        'tense': sorted(df['tense'].unique().tolist()),
        'certainty': sorted(df['certainty'].unique().tolist())
    }

# =========================
# Cues / Prompt
# =========================
def top_tokens_by_class(
    texts, labels, token_pattern, ngram_range, min_df, top_k=10, neutral_band=0.15
):
    # 1) 정규화 텍스트 사용
    norm_texts = [normalize_text(t) for t in texts]

    vect = CountVectorizer(
        analyzer="word",
        token_pattern=token_pattern,
        ngram_range=ngram_range,   # (cfg.MIN_N, cfg.MAX_N)도 가능
        min_df=min_df,
        lowercase=False,
    )
    X = vect.fit_transform(norm_texts)
    vocab = np.array(vect.get_feature_names_out())

    # 2) 전역 불용/잡음 제거
    STOP_SET = {"이다","한다","했다","있다","것","등","및","에서","으로","대한","관련","수","로","가","은","는","을","를"}
    dfreq = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = (np.array([w not in STOP_SET and len(w) > 1 for w in vocab]))
    X = X[:, keep]; vocab = vocab[keep]

    classes = list(pd.Series(labels).unique())
    res = {}
    V = len(vocab); alpha = 0.5
    for c in classes:
        y = (np.array(labels) == c)
        c1 = np.asarray(X[y].sum(axis=0)).ravel()
        c0 = np.asarray(X[~y].sum(axis=0)).ravel()
        p1 = (c1 + alpha) / (c1.sum() + alpha * V)
        p0 = (c0 + alpha) / (c0.sum() + alpha * V)
        lr = np.log(p1 / p0)

        # 3) 클래스-중립 토큰 제거 (|lr|가 작은 토큰 컷)
        sel = np.where(np.abs(lr) > neutral_band)[0]
        order = sel[np.argsort(-lr[sel])]

        toks = []
        for i in order:
            w = vocab[i]
            toks.append(w)
            if len(toks) >= top_k:
                break
        res[c] = toks
    return res

def cue_block_by_label(title: str, cues: Dict[str, List[str]], take=10) -> str:
    def fmt_tok(t: str) -> str:
        return f"`{t}`" if t.startswith("<") and t.endswith(">") else t

    lines = []
    for label in cues:
        toks = (cues.get(label) or [])[:take]
        toks = [fmt_tok(t) for t in toks]
        lines.append(f"- {label}: {', '.join(toks)}")
    return f"[{title} 단서]\n" + "\n".join(lines)

def build_prompt(df: pd.DataFrame,
                 max_chars=cfg.PROMPT_MAX_CHARS,
                 n_examples=cfg.EXAMPLES_PER_PROMPT,
                 seed: int = 197) -> str:
    type_set  = sorted(df['type'].unique().tolist())
    pol_set   = sorted(df['polarity'].unique().tolist())
    tense_set = sorted(df['tense'].unique().tolist())
    cert_set  = sorted(df['certainty'].unique().tolist())

    # 단서 추출(정규화 기반)
    type_cues  = top_tokens_by_class(df['user_prompt'], df['type'],      cfg.TOKEN_PATTERN, (cfg.MIN_N,cfg.MAX_N), cfg.MIN_DF, top_k=10)
    pol_cues   = top_tokens_by_class(df['user_prompt'], df['polarity'],  cfg.TOKEN_PATTERN, (cfg.MIN_N,cfg.MAX_N), cfg.MIN_DF, top_k=10)
    tense_cues = top_tokens_by_class(df['user_prompt'], df['tense'],     cfg.TOKEN_PATTERN, (cfg.MIN_N,cfg.MAX_N), cfg.MIN_DF, top_k=10)
    cert_cues  = top_tokens_by_class(df['user_prompt'], df['certainty'], cfg.TOKEN_PATTERN, (cfg.MIN_N,cfg.MAX_N), cfg.MIN_DF, top_k=10)


    # 강화 헤더(간결+정규화+우선순위)
    header = (
f"""### 역할 및 목표
주어진 한국어 문장을 4가지 속성(유형, 극성, 시제, 확실성)으로 정확히 분류하라. \
오탈자·띄어쓰기 오류는 의미를 보정하여 해석하고 아래 규칙을 엄격히 따를 것.

### 라벨(데이터 기반·반드시 이 집합만 사용)
- 유형: {{{', '.join(type_set)}}}
- 극성: {{{', '.join(pol_set)}}}
- 시제: {{{', '.join(tense_set)}}}
- 확실성: {{{', '.join(cert_set)}}}

### 정규화 규칙(의미 토큰)
- `\u003cFUTURE_PLAN\u003e`: 할 예정이다/계획/될 전망/될 것/되겠다/있겠다/예상/전망/목표/추진…/선보일 예정/출시할 예정
- `\u003cSCHEDULE_NOW\u003e`: 출시한다/선보인다/개최한다/오픈한다/진행한다(일정형 현재 → 미래 간주)
- `\u003cPAST_REPORT\u003e`: 발표했다/밝혔다/전했다/기록했다/했다/였다/되었다/…
- `\u003cINTERACT\u003e`: ?/십시오/하세요/합시다/요청/권고/주의/안내
- `\u003cNEG\u003e`: 아니다/않/못/없/부진/감소/하락/불가/불가능
- `\u003cUNCERT\u003e`: 수 있다/가능성/전망/예상/의도/계획/추정/관측/것 같다/보인다/듯하다

### 우선순위 분류 절차
1) 유형 즉시판정: `\u003cINTERACT\u003e`가 있으면 유형=대화형.
2) 시제: `\u003cPAST_REPORT\u003e`→과거, `\u003cFUTURE_PLAN\u003e` 또는 `\u003cSCHEDULE_NOW\u003e`→미래, 그 외 현재.
3) 극성: `\u003cNEG\u003e`→부정, 애매하면 미정, 그 외 긍정.
4) 확실성: `\u003cUNCERT\u003e`→불확실, 그 외 확실.
5) 유형(최종): (1번 아닌 경우)
   - 미래 의도/전망이 핵심이면 예측형
   - 근거+추정/분석이면 추론형
   - 사실/수치/보고 중심이면 사실형

### 충돌 해소
- `\u003cPAST_REPORT\u003e` + ‘전망/예상’: 시제=과거, 확실성=불확실, 유형=사실형(‘전망을 발표했다’는 사실 보고).
- ‘가능성’ + `\u003cNEG\u003e`: 극성=부정, 확실성=확실(부정을 단정 보고).

### 출력 형식(필수)
- `번호.유형,극성,시제,확실성` 만 출력(공백/추가기호 금지).
"""
    )
    
    guides = "\n\n".join([
        cue_block_by_label("유형", type_cues, 10),
        cue_block_by_label("극성", pol_cues, 10),
        cue_block_by_label("시제", tense_cues, 10),
        cue_block_by_label("확실성", cert_cues, 10),
    ])

    # 예시
    rng = np.random.RandomState(seed)
    ex_df = df.sample(min(n_examples, len(df)), random_state=rng)
    ex_block = "[예시]\n" + "\n".join(
        f"{i+1}. 문장: {r.user_prompt}\n{i+1}. {r.type},{r.polarity},{r.tense},{r.certainty}"
        for i, r in enumerate(ex_df.itertuples(index=False))
    )

    return (header + "\n" + guides + "\n\n" + ex_block)[:max_chars]

# =========================
# Rule features / predictor
# =========================
def create_rule_features(texts: List[str]) -> csr_matrix:
    rows = []
    for raw in texts:
        t = normalize_text(str(raw))

        # ---- NEW: token-based tense
        tense = _tense_label_from_norm(t)
        past    = 1 if tense == "과거"  else 0
        future  = 1 if tense == "미래"  else 0
        present = 1 if tense == "현재"  else 0

        # 기타 신호
        interact = 1 if TYPE_INTERACT.search(t) else 0
        predict  = 1 if TYPE_PREDICT.search(t)  else 0
        infer    = 1 if TYPE_INFER.search(t)    else 0
        neg      = 1 if POL_NEG.search(t)       else 0
        uncert   = 1 if UNCERTAIN.search(t)     else 0

        rows.append([past, present, future, interact, predict, infer, neg, uncert])
    return csr_matrix(np.asarray(rows, dtype=np.float32))

# =========================
# Boosting (optional)
# =========================
def predict_with_boosting(df_train: pd.DataFrame, df_test: pd.DataFrame,
                          label_sets: Dict[str, List[str]], cfg: Config) -> Dict[str, List[str]]:
    if not HAS_LGB:
        raise RuntimeError("lightgbm 미설치: pip install lightgbm")

    preds = {key: [] for key in label_sets.keys()}

    # 공통 피처: 정규화 텍스트를 사용
    vec = TfidfVectorizer(token_pattern=cfg.TOKEN_PATTERN,
                          stop_words=cfg.STOP_WORDS,
                          ngram_range=(1,3), min_df=2)
    X_train_vec = vec.fit_transform(df_train['user_prompt'].map(normalize_text))
    X_test_vec  = vec.transform(df_test['user_prompt'].map(normalize_text))
    X_train_rules = create_rule_features(df_train['user_prompt'].tolist())
    X_test_rules  = create_rule_features(df_test['user_prompt'].tolist())
    X_train = hstack([X_train_vec, X_train_rules])
    X_test  = hstack([X_test_vec,  X_test_rules])

    for label_type, labels in label_sets.items():
        if len(labels) < 2:
            preds[label_type] = [labels[0]] * len(df_test)
            continue
        y_train = df_train[label_type].values
        clf = lgb.LGBMClassifier(**{**cfg.BOOSTING_PARAMS, "num_class": len(labels)})
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        preds[label_type] = y_pred.tolist()
    return preds

# =========================
# Post override (규칙 보정)
# =========================
def _map_label(preferred: str, fallbacks: List[str], available: List[str]) -> str:
    for cand in [preferred] + fallbacks:
        if cand in available:
            return cand
    return available[0]

def post_override(text: str, type_pred: str, tense_pred: str,
                  type_labels: List[str], tense_labels: List[str]) -> Tuple[str, str]:
    """정규화 토큰 기준 사후 보정: 시제는 helper로 재산출, 유형은 상호작용/예측/추론 우선순위 적용."""
    t = normalize_text(text)

    # ---- NEW: 일관된 토큰기반 시제 재평가
    new_tense = _tense_label_from_norm(t)

    # 유형 우선순위 보정
    if TYPE_INTERACT.search(t):
        new_type = _map_label("대화형", ["추론형", "사실형", "예측형"], type_labels)
    elif TYPE_PREDICT.search(t) and type_pred != "대화형":
        new_type = _map_label("예측형", ["추론형", "사실형"], type_labels)
    elif TYPE_INFER.search(t) and type_pred not in ("대화형", "예측형"):
        new_type = _map_label("추론형", ["사실형"], type_labels)
    else:
        new_type = type_pred

    # 라벨셋 밖 값 방지
    new_tense = _map_label(new_tense, ["현재", "과거", "미래"], tense_labels)
    return new_type, new_tense

# =========================
# I/O
# =========================
def load_numbered_sentences(file_path: Path) -> List[Tuple[int, str]]:
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
            if m:
                items.append((int(m.group(1)), m.group(2)))
            else:
                items.append((len(items)+1, line))
    return items

def make_header(score: float, ts: str,
                field_acc: Dict[str, float],
                label_counts: Dict[str, Dict[str, int]],
                n_samples: int) -> str:
    model_name = "LightGBM" if cfg.USE_BOOSTING else "Rule"
    lines = []
    lines.append(f"# SCORE={score:.4f}")
    lines.append(f"# TIMESTAMP={ts}")
    lines.append(f"# MODEL={model_name}")
    lines.append(f"# N_SAMPLES={n_samples}  KFOLDS={cfg.N_SPLITS}")
    lines.append("# ACC " + " ".join([f"{k}={field_acc.get(k,0.0):.4f}" for k in ["type","polarity","tense","certainty"]]))
    for f in ["type","polarity","tense","certainty"]:
        cnt = label_counts.get(f, {})
        cnt_str = ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        lines.append(f"# {f.upper()}_COUNTS {cnt_str}")
    lines.append(f"# CONFIG USE_RULE_OVERRIDES={cfg.USE_RULE_OVERRIDES} TOKEN_PATTERN={cfg.TOKEN_PATTERN} NGRAM={cfg.MIN_N}-{cfg.MAX_N} MIN_DF={cfg.MIN_DF}")
    return "\n".join(lines)

# =========================
# Eval (OOF)
# =========================
def run_eval(df: pd.DataFrame) -> tuple[float, Dict[str, float], Dict[str, Dict[str, int]]]:
    fields = ["type","polarity","tense","certainty"]
    label_counts = {f: df[f].value_counts().to_dict() for f in fields}
    n_splits = min(cfg.N_SPLITS, max(2, len(df)//2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs = {f: [] for f in fields}
    for tr, va in kf.split(df):
        tr_df = df.iloc[tr].reset_index(drop=True)
        va_df = df.iloc[va].reset_index(drop=True)
        labels_tr = build_label_sets(tr_df)

        if cfg.USE_BOOSTING:
            preds = predict_with_boosting(tr_df, va_df[['user_prompt']].copy(), labels_tr, cfg)
        else:
            preds = predict_with_rules(va_df[['user_prompt']].copy(), labels_tr)
            if cfg.USE_RULE_OVERRIDES:
                type_labels  = labels_tr['type']
                tense_labels = labels_tr['tense']
                va_texts = va_df["user_prompt"].tolist()
                for i, txt in enumerate(va_texts):
                    nt, nz = post_override(txt, preds["type"][i], preds["tense"][i], type_labels, tense_labels)
                    preds["type"][i]  = nt
                    preds["tense"][i] = nz

        for f in fields:
            accs[f].append( (pd.Series(preds[f]) == va_df[f].values).mean() )

    field_acc = {f: float(np.mean(v)) for f, v in accs.items()}
    macro_acc = float(np.mean(list(field_acc.values())))
    print("[SCORE]", " ".join([f"{k}={field_acc[k]:.4f}" for k in fields]), f"| macro={macro_acc:.4f}")
    sys.stdout.flush()
    return macro_acc, field_acc, label_counts

# =========================
# Main
# =========================
def main():
    # 1) 데이터 로드
    df = load_data(cfg)
    labels = build_label_sets(df)

    # 2) 프롬프트 생성(강화 헤더+단서+예시) — 저장은 평가 후 점수/타임스탬프 반영
    prompt = build_prompt(df, max_chars=cfg.PROMPT_MAX_CHARS, n_examples=cfg.EXAMPLES_PER_PROMPT, seed=197)

    # 3) OOF 평가
    score, field_acc, label_counts = run_eval(df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    header_block = make_header(score, ts, field_acc, label_counts, len(df))

    # 4) 프롬프트 저장 (파일명에 score+timestamp)
    out_prompt = Path(cfg.SAVE_PATH) / f"system_prompt_{score:.4f}_{ts}.txt"
    with open(out_prompt, "w", encoding="utf-8") as f:
        f.write(header_block + "\n\n")
        f.write(prompt + "\n")
    print(f"[OK] 시스템 프롬프트 저장: {out_prompt}")

    # 5) (옵션) 채점 입력 예측
    test_in = Path(cfg.PATH) / "test_input.txt"
    if test_in.exists():
        items = load_numbered_sentences(test_in)
        texts = [t for _, t in items]

        if cfg.USE_BOOSTING:
            preds = predict_with_boosting(df, pd.DataFrame({'user_prompt': texts}), labels, cfg)
        else:
            preds = predict_with_rules(pd.DataFrame({'user_prompt': texts}), labels)
            # 부스팅/규칙 공통 사후 보정
            if cfg.USE_RULE_OVERRIDES:
                type_labels  = labels['type']
                tense_labels = labels['tense']
                for i, txt in enumerate(texts):
                    nt, nz = post_override(txt, preds["type"][i], preds["tense"][i],
                                        type_labels, tense_labels)
                    preds["type"][i]  = nt
                    preds["tense"][i] = nz


        out_lines = [
            f"{idx}.{t},{p},{s},{c}"   # ← 점 뒤 공백 제거
            for (idx,_), t,p,s,c in zip(items,
                                        preds["type"],
                                        preds["polarity"],
                                        preds["tense"],
                                        preds["certainty"])
        ]


        out_file = Path(cfg.SAVE_PATH) / f"submission_{score:.4f}_{ts}.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(header_block + "\n")
            for ln in out_lines:
                f.write(ln + "\n")
        print(f"[OK] 예측 저장: {out_file}")

if __name__ == "__main__":
    main()
