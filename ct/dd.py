import os
import sys
import argparse
import datetime
import pandas as pd

# ------------------------------
# 1) 입력 경로 처리
# ------------------------------
ap = argparse.ArgumentParser()
ap.add_argument(
    "--csv",
    default="./_data/dacon/t/samples.csv",
    help="평가용 CSV 경로 (기본: /mnt/data/samples.csv)",
)
args = ap.parse_args()

CSV_PATH = args.csv
if not os.path.exists(CSV_PATH):
    print(f"[ERROR] CSV not found: {CSV_PATH}")
    sys.exit(1)

# ------------------------------
# 2) 유저 프롬프트 빌더 (플랫폼 포맷)
# ------------------------------
def build_user_prompt(title: str, content: str) -> str:
    return f"[기사]\n\n제목: {title}\n\n내용: {content}"

# ------------------------------
# 3) 시스템 프롬프트(업로드용) 생성
# ------------------------------
SYSTEM_PROMPT = """역할: 당신은 자동차 관련 뉴스 여부를 0/1로 분류하는 필터입니다.
중요: 출력은 오직 하나의 숫자만 허용됩니다. 정확히 '0' 또는 '1' 한 글자.
금지: 설명, 접두/접미 텍스트, 따옴표, 공백, 줄바꿈, 마크다운, 코드블록, 이유, 토큰 카운트, 확률 등 일절 금지.
판정 실패나 모호함이 있으면 '0'만 출력합니다.

입력 형식:
[기사]

제목: <title>
내용: <content>

과제:
- '해당 기사가 자동차(automotive)와 직접적으로 관련 있는가?'를 판정합니다.
- 자동차 관련성은 다음 중 하나 이상이 명확히 드러나면 '1', 아니면 '0'입니다.

포함(1) 기준 예시(하나라도 충족 시 1):
- 완성차/부품/기술/산업/정책/시장 동향(전기차, 자율주행, 배터리-자동차 맥락, 충전 인프라, 모빌리티 서비스 등)
- 자동차 제조사·브랜드: 현대차, 기아, 제네시스, 쌍용, 르노코리아, 한국지엠, 테슬라, 도요타, 혼다, 닛산, BYD, BMW, 벤츠, 아우디, 폭스바겐, 포드, GM 등
- 부품/기술/규제: 엔진, 모터, 샤시, ECU, ADAS, 라이다, OTA, 충전소, 배터리 셀·팩·BMS(자동차 맥락), 연비·배출가스·안전규제 등
- 상용차/이륜차/자전거가 자동차 산업 맥락에서 다루어지는 경우
- 자동차 산업의 투자/실적/리콜/생산/수출입/노사 등

제외(0) 기준 예시(하나라도 충족 시 0):
- '현대'/'기아'가 자동차사가 아닌 일반적 의미로 쓰인 경우(예: 현대 사회, 현대 미술, 기아 문제)
- '배터리'가 스마트폰/가전/ESS 등 자동차 맥락이 전혀 없는 경우
- 운전 은유, 스포츠 팀명(현대 모터스포츠가 아닌 경우), 소설·영화 속 은유적 자동차 언급
- 일반 교통 이슈(대중교통, 항공/철도/선박)만 다루고 자동차 산업 맥락이 없는 경우
- 광고/스팸/가격비교 글 등 실질적 뉴스가 아닌 경우

절차:
1) 제목과 내용을 모두 확인합니다(한쪽이 비어도 가능한 범위에서 판정).
2) 제외(0) 조건을 먼저 검사합니다. 만족하면 즉시 '0' 결정.
3) 포함(1) 조건의 뚜렷한 증거가 하나 이상 있으면 '1' 결정.
4) 자동차 맥락이 불명확·모호하면 보수적으로 '0' 결정.

최종 출력 규칙(매우 중요):
- 반드시 숫자 하나만 출력: 0 또는 1. 공백/개행/따옴표/텍스트 금지.
- 생각, 근거, 단계, 이유, 요약을 출력하지 마세요.
- 규칙을 어겼다고 판단되면 즉시 '0'만 출력.

이제 기사에 대해 최종 한 글자(0 또는 1)만 출력하세요.""".strip()

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PROMPT_PATH = f"./dacon_system_prompt_{ts}.txt"
with open(OUT_PROMPT_PATH, "w", encoding="utf-8") as f:
    f.write(SYSTEM_PROMPT + "\n")
print(f"[OK] 시스템 프롬프트 파일 저장: {OUT_PROMPT_PATH}")

# ------------------------------
# 4) 로컬 평가 (MOCK → OPENAI 순으로 시도)
# ------------------------------
def eval_mock(df: pd.DataFrame) -> float:
    AUTO_POS = [
        "자동차","모빌리티","전기차","ev","자율주행","하이브리드","연비","리콜","차량","세단","suv",
        "현대차","기아","제네시스","쌍용","르노코리아","한국지엠","테슬라","도요타","혼다","닛산","bmw","벤츠",
        "아우디","폭스바겐","포드","gm","byd","충전소","배터리","bms","셀","팩","모터","엔진","변속기","샤시",
        "adas","라이다","lidar","ecu","ota","타이어","브레이크","리세일","출고","판매량","생산","공장","완성차"
    ]
    AUTO_NEG = ["현대 미술","현대 사회","아동 기아","국제 기아","핸드폰 배터리","휴대폰 배터리"]

    def judge(text: str) -> str:
        t = text.lower()
        if any(neg.lower() in t for neg in AUTO_NEG):
            return "0"
        hit = any(pos.lower() in t for pos in AUTO_POS)
        return "1" if hit else "0"

    correct, n = 0, 0
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        y = str(int(row.get("label", 0)))
        up = build_user_prompt(title, content)
        yhat = judge(up)
        correct += int(yhat == y)
        n += 1
    return (correct / n) if n else 0.0

def eval_openai(df: pd.DataFrame, system_prompt: str) -> float:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return 0.0
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        return 0.0

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    correct, n = 0, 0
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        y = str(int(row.get("label", 0)))
        up = build_user_prompt(title, content)
        try:
            resp = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=up,
                temperature=0,
                max_output_tokens=1,
                store=False,
            )
            text = (resp.output_text or "").strip()
            c = text.replace("\n", " ").replace("\r", " ")
            yhat = c[0] if c and c[0] in ("0", "1") else "0"
        except Exception:
            yhat = "0"
        correct += int(yhat == y)
        n += 1
    return (correct / n) if n else 0.0

# 데이터 로드
df = pd.read_csv(CSV_PATH)

# MOCK
acc_mock = eval_mock(df)
print(f"[MOCK] Local ACC: {acc_mock:.4f}")

# OPENAI(선택)
acc_openai = eval_openai(df, SYSTEM_PROMPT)
if acc_openai > 0:
    print(f"[OPENAI] Local ACC: {acc_openai:.4f}")
else:
    print("[OPENAI] API 키/SDK 미설치 또는 호출 실패로 스킵했습니다. (OPENAI_API_KEY 설정 시 평가 가능)")

# ------------------------------
# 5) 제출 안내
# ------------------------------
print(
    "\n[제출 안내]\n"
    "- 데이콘 제출물은 '시스템 프롬프트 본문'만 업로드합니다.\n"
    "- 방금 저장된 파일을 열어 내용 전체를 복사하여 제출하세요.\n"
    "- 출력은 정확히 한 글자(0 또는 1)만 나오도록 프롬프트에서 엄격히 규정했습니다.\n"
)
