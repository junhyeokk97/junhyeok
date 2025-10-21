#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re, pandas as pd, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

# === 네가 쓰던 토큰 패턴(AN) 그대로 사용 ===
TOKEN_PATTERN_AN = r"(?u)(?:\b[가-힣A-Za-z]{2,}\b|\b\d{1,3}%\b|\b\d{4}\b|\b\d+(?:\.\d+)?\b)"

def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F\u200B-\u200F\uFEFF]", "", s)
    s = "\n".join(re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in s.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip()

def keyword_counts_from_df(df: pd.DataFrame,
                           token_pattern: str = TOKEN_PATTERN_AN,
                           min_df: int = 1,
                           ngram_max: int = 3) -> pd.DataFrame:
    """
    title+content를 합쳐서 전체 키워드/라벨별 빈도(TF)와 문서수(DF)를 모두 계산.
    또한 title만, content만의 빈도도 함께 제공.
    반환: 키워드별 집계 DataFrame
    """
    # 1) 텍스트 준비
    titles = df["title"].fillna("").astype(str).apply(clean_text)
    contents = df["content"].fillna("").astype(str).apply(clean_text)
    texts_all = (titles + "\n" + contents).tolist()

    # 2) 벡터라이저(전체 코퍼스 기준 vocab 고정)
    vect = CountVectorizer(
        analyzer="word",
        token_pattern=token_pattern,
        ngram_range=(1, ngram_max),
        min_df=min_df
    )
    X_all = vect.fit_transform(texts_all)   # 전체 문서
    vocab = np.asarray(vect.get_feature_names_out())

    # 3) 라벨 마스크
    if "label" in df.columns:
        y = df["label"].astype(int).to_numpy()
        idx_pos = (y == 1)
        idx_neg = (y == 0)
    else:
        # 라벨 없으면 전부를 1로 취급 (필요시 조정)
        idx_pos = np.ones(len(df), dtype=bool)
        idx_neg = np.zeros(len(df), dtype=bool)

    # 4) 전체/라벨별 TF(합계) & DF(문서수)
    def _tf(m):  return np.asarray(m.sum(axis=0)).ravel()
    def _df(m):  return np.asarray((m > 0).sum(axis=0)).ravel()

    tf_all = _tf(X_all)
    df_all = _df(X_all)

    X_pos = X_all[idx_pos] if idx_pos.any() else X_all[:0]
    X_neg = X_all[idx_neg] if idx_neg.any() else X_all[:0]

    tf_pos = _tf(X_pos) if idx_pos.any() else np.zeros_like(tf_all)
    df_pos = _df(X_pos) if idx_pos.any() else np.zeros_like(df_all)

    tf_neg = _tf(X_neg) if idx_neg.any() else np.zeros_like(tf_all)
    df_neg = _df(X_neg) if idx_neg.any() else np.zeros_like(df_all)

    # 5) title 전용 / content 전용 TF
    X_title = vect.transform(titles.tolist())
    X_content = vect.transform(contents.tolist())

    tf_title_all = _tf(X_title)
    tf_content_all = _tf(X_content)

    # 6) 결과 테이블
    out = pd.DataFrame({
        "token": vocab,
        "tf_all": tf_all,
        "df_all": df_all,
        "tf_pos": tf_pos,
        "df_pos": df_pos,
        "tf_neg": tf_neg,
        "df_neg": df_neg,
        "tf_title_all": tf_title_all,
        "tf_content_all": tf_content_all,
    }).sort_values(["tf_all", "df_all"], ascending=[False, False]).reset_index(drop=True)

    return out

if __name__ == "__main__":
    # 파일 경로 (업로드한 경로)
    csv_path = Path("/mnt/data/samples.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    counts = keyword_counts_from_df(
        df,
        token_pattern=TOKEN_PATTERN_AN,
        min_df=1,          # 희귀어도 다 보고 싶으면 1
        ngram_max=3        # 유니/바이/트라이그램까지
    )

    # 상위 50개 샘플 확인
    print(counts.head(50))

    # 저장 (원하면 엑셀/CSV 모두)
    out_csv = csv_path.parent / "keyword_counts.csv"
    counts.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 저장: {out_csv}")
