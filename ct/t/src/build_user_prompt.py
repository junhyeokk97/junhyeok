# -*- coding: utf-8 -*-
"""
입력 정규화/클린업을 거쳐 대회 포맷으로 유저 프롬프트를 생성.
- 제어문자/제로폭 문자 제거, 개행/공백 정규화, HTML 태그/이모지 제거(선택), 길이 캡(선택)
- 시그니처는 그대로: build_user_prompt(title, content)
"""
from __future__ import annotations
import re
from typing import Optional

# === 설정(필요시 조절) ===
STRIP_HTML = True            # HTML 태그 제거
STRIP_EMOJI = True           # 이모지 제거
MAX_TITLE_CHARS = 500        # 제목 최대 길이(0=무제한)
MAX_CONTENT_CHARS = 12000    # 본문 최대 길이(0=무제한)
TITLE_PLACEHOLDER = "(제목 없음)"
CONTENT_PLACEHOLDER = ""     # 본문이 비면 빈 문자열 유지(권장)

# 제어문자/제로폭 문자
_CTRL_ZERO_WIDTH = re.compile(
    r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F\u200B-\u200F\uFEFF]"
)
# 여러 개의 공백 → 한 칸, \r\n → \n
_WS_NORM = re.compile(r"[ \t\f\v]+")
# 3개 이상 연속 개행 → 2개
_MULTI_NL = re.compile(r"\n{3,}")
# 간단 HTML 태그 제거
_HTML_TAG = re.compile(r"<[^>]+>")
# 이모지(대략적)
_EMOJI = re.compile(
    r"[\U0001F1E6-\U0001F1FF]|"  # flags
    r"[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
    r"[\U0001F600-\U0001F64F]|"  # emoticons
    r"[\U0001F680-\U0001F6FF]|"  # transport & map
    r"[\U0001F700-\U0001F77F]|"
    r"[\U0001F780-\U0001F7FF]|"
    r"[\U0001F800-\U0001F8FF]|"
    r"[\U0001F900-\U0001F9FF]|"
    r"[\U0001FA00-\U0001FA6F]|"
    r"[\U0001FA70-\U0001FAFF]"
)

def _to_text(x: Optional[object]) -> str:
    if x is None:
        return ""
    return str(x)

def _clean(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL_ZERO_WIDTH.sub("", s)
    if STRIP_HTML:
        s = _HTML_TAG.sub("", s)
    if STRIP_EMOJI:
        s = _EMOJI.sub("", s)
    # 라인 내부 공백 정규화
    s = "\n".join(_WS_NORM.sub(" ", ln).strip() for ln in s.split("\n"))
    # 과도한 연속 개행 축소
    s = _MULTI_NL.sub("\n\n", s)
    return s.strip()

def _clip(s: str, limit: int) -> str:
    if limit and len(s) > limit:
        return s[: max(0, limit - 1)] + "…"
    return s

def build_user_prompt(title: Optional[object], content: Optional[object]) -> str:
    """대회 입력 포맷을 정확히 생성."""
    t = _clip(_clean(_to_text(title)), MAX_TITLE_CHARS) or TITLE_PLACEHOLDER
    c = _clip(_clean(_to_text(content)), MAX_CONTENT_CHARS) or CONTENT_PLACEHOLDER
    return f"[기사]\n\n제목: {t}\n\n내용: {c}"
