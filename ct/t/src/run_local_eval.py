# src/run_local_eval.py
import os, sys, time, datetime, pandas as pd
from typing import Iterable
sys.path.append(os.path.dirname(__file__))
from build_user_prompt import build_user_prompt

# 시스템 프롬프트 로드
def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 로컬 점검용 더미 분류기 (키워드 규칙 기반) — 실제 대회 점수와 무관
AUTO_POS = [
    "자동차","모빌리티","전기차","ev","자율주행","하이브리드","연비","리콜","차량","세단","suv",
    "현대차","기아","제네시스","쌍용","르노코리아","한국지엠","테슬라","도요타","혼다","닛산","bmw","벤츠",
    "아우디","폭스바겐","포드","gm","byd","충전소","배터리","bms","셀","팩","모터","엔진","변속기","샤시",
    "adas","라이다","lidar","ecu","ota","타이어","브레이크","리세일","출고","판매량","생산","공장","완성차"
]
AUTO_NEG = [
    # 자동차사가 아닌 일반문맥/동음이의어 상황을 일부 차단
    "현대 미술","현대 사회","아동 기아","국제 기아","배터리 잔량(휴대폰)","핸드폰 배터리"
]

def mock_llm(system_prompt: str, user_prompt: str) -> str:
    text = f"{system_prompt}\n{user_prompt}".lower()
    if any(neg.lower() in text for neg in AUTO_NEG):
        return "0"
    hit = any(pos.lower() in text for pos in AUTO_POS)
    return "1" if hit else "0"

def evaluate(csv_path: str, system_prompt_path: str) -> float:
    df = pd.read_csv(csv_path)
    sys_prompt = load_system_prompt(system_prompt_path)
    preds = []
    for _, row in df.iterrows():
        title = str(row.get("title",""))
        content = str(row.get("content",""))
        y = int(row.get("label", 0))
        up = build_user_prompt(title, content)
        yhat = mock_llm(sys_prompt, up).strip()
        yhat = 1 if yhat == "1" else 0
        preds.append(yhat == y)
    acc = sum(preds) / len(preds) if len(preds) else 0.0
    # 로깅
    logdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exp"))
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "runs.csv"), "a", encoding="utf-8") as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{now},v001_system.txt,{acc:.6f},{os.path.basename(csv_path)}\n")
    return acc

if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_csv = os.path.join(base, "data", "samples.csv")
    sys_prompt_path = os.path.join(base, "prompts", "v001_system.txt")
    acc = evaluate(data_csv, sys_prompt_path)
    print(f"[MOCK] Local ACC: {acc:.4f}")
