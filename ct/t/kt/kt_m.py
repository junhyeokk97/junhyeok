#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
import re
import datetime

# --- 1. 기본 설정 ---
class Config:
    """스크립트의 경로 및 기본 설정을 관리합니다."""
    DATA_PATH = 'C:/vscode_py/_data/dacon/kt/'
    CSV_NAME = "samples.csv"
    SAVE_PATH = 'C:/vscode_py/_save/dacon/kt/'
    OUTPUT_PREFIX = "kt_system_prompt"
    
cfg = Config()
os.makedirs(cfg.SAVE_PATH, exist_ok=True)

# --- 2. 고도화된 시스템 프롬프트 템플릿 ---
SYSTEM_PROMPT_TEMPLATE = """
### 역할 및 목표
당신은 주어진 한국어 문장을 4가지 기준에 따라 분류하는 AI 전문가입니다. 각 문장을 '유형', '극성', '시제', '확실성'의 4가지 속성으로 분류하고, 반드시 지정된 출력 형식에 맞춰 응답해야 합니다.

### 분류 기준
1.  **유형**: 사실형, 추론형, 대화형, 예측형
2.  **극성**: 긍정, 부정, 미정
3.  **시제**: 과거, 현재, 미래
4.  **확실성**: 확실, 불확실

### 사고 절차 (Step-by-Step)
1.  먼저 문장의 핵심 동사와 시제 관련 표현(예: '했다', '것이다')을 찾아 **시제**를 결정합니다.
2.  다음으로, 문장에 부정 표현('안', '못', '없다' 등)이 있는지 확인하여 **극성**을 결정합니다.
3.  그 다음, 추측이나 가능성을 나타내는 표현('같다', '수도 있다', '전망')이 있는지에 따라 **확실성**을 판단합니다.
4.  마지막으로, 위 정보를 종합하고 문장 구조(평서문, 의문문)를 고려하여 최종 **유형**을 결정합니다.

### 분류 예시 (Few-Shot Examples)
아래는 당신이 따라야 할 모범적인 분류 예시입니다. 이 패턴을 학습하여 동일한 방식으로 결과를 도출하십시오.
{few_shot_examples}

### 참고 키워드 (데이터 분석 기반)
아래는 각 분류 라벨에서 주로 나타나는 핵심 키워드입니다. 판단이 어려울 때 참고하십시오.
{keyword_hints}

### 최종 출력 형식 (매우 중요)
-   `번호.유형,극성,시제,확실성` 형식으로만 응답해야 합니다.
-   쉼표(,) 외에 다른 공백이나 특수문자는 절대 포함하지 마십시오.
"""

# --- 3. 데이터 분석 및 프롬프트 생성 함수 ---

def analyze_and_prepare_data(df: pd.DataFrame, top_n_keywords: int = 10) -> (dict, str):
    """
    데이터를 분석하여 '핵심 키워드'와 '모범 예제(Few-Shot)'를 모두 추출합니다.
    """
    print("데이터 분석 및 학습 자료(키워드, 예시) 생성 시작...")
    
    # 'output' 컬럼을 4개의 속성 컬럼으로 분리
    labels_df = df['output'].str.split(',', expand=True)
    labels_df.columns = ['유형', '극성', '시제', '확실성']
    df_processed = pd.concat([df['user_prompt'], labels_df, df['output']], axis=1)

    # 속성별 키워드 추출
    attribute_keywords = {}
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[가-힣]{2,}\b", max_features=2000)
    vectorizer.fit(df_processed['user_prompt'])
    
    for attribute in tqdm(['유형', '극성', '시제', '확실성'], desc="속성별 키워드 분석"):
        attribute_keywords[attribute] = {}
        for label, group in df_processed.groupby(attribute):
            corpus = " ".join(group['user_prompt'])
            tfidf_matrix = vectorizer.transform([corpus])
            
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray().flatten()
            top_indices = scores.argsort()[-top_n_keywords:]
            top_keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            attribute_keywords[attribute][label] = top_keywords

    # 유형별 대표 예시(Few-Shot) 자동 선택
    few_shot_examples = []
    for sentence_type in ['사실형', '추론형', '예측형', '대화형']:
        subset = df_processed[df_processed['유형'] == sentence_type]
        if not subset.empty:
            best_example = subset.loc[subset['user_prompt'].str.len().idxmin()]
            example_sentence = best_example['user_prompt']
            example_label = best_example['output']
            few_shot_examples.append(f"- 입력: \"{example_sentence}\"\n- 출력: {example_label}")

    print("학습 자료 생성 완료!")
    return attribute_keywords, "\n".join(few_shot_examples)


def assemble_final_prompt(template: str, keyword_db: dict, examples: str) -> str:
    """
    모든 분석 결과를 템플릿에 결합하여 최종 프롬프트를 완성합니다.
    """
    hints = []
    for attribute, labels in keyword_db.items():
        hints.append(f"#### {attribute}")
        for label, keywords in labels.items():
            keyword_str = ", ".join(keywords)
            hints.append(f"- **{label}**: {keyword_str}")
        hints.append("")
    
    keyword_hints_str = "\n".join(hints)
    prompt = template.format(few_shot_examples=examples, keyword_hints=keyword_hints_str)
    return prompt

# --- 4. 메인 실행 로직 ---
if __name__ == "__main__":
    
    try:
        input_path = Path(cfg.DATA_PATH) / cfg.CSV_NAME
        df = pd.read_csv(input_path)
        print(f"'{input_path}' 에서 {len(df):,}개의 샘플을 로드했습니다.")
        
        keyword_database, few_shot_examples_str = analyze_and_prepare_data(df)
        final_prompt = assemble_final_prompt(SYSTEM_PROMPT_TEMPLATE, keyword_database, few_shot_examples_str)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_filename = f"{cfg.OUTPUT_PREFIX}_{timestamp}.txt"
        out_path = Path(cfg.SAVE_PATH) / out_filename
        out_path.write_text(final_prompt, encoding="utf-8")
        
        print(f"\n[OK] 최종 시스템 프롬프트가 성공적으로 생성되었습니다: {out_path}")
        print("\n이제 이 파일을 열어 내용을 복사하여 GPT-4o의 시스템 프롬프트로 사용하시면 됩니다.")

    except FileNotFoundError:
        print(f"오류: 지정된 경로에 파일이 없습니다. 경로를 다시 확인해주세요.\n-> {input_path}")
    except Exception as e:
        print(f"코드 실행 중 오류가 발생했습니다: {e}")