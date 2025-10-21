################
"""
1. 시작 - 환경 - 시스템환경 변수편집 - 계정의 환경 변수 편집
2. 사용자 변수에 '새로만들기'
3. 변수이름 : OPENAI_API_KEY
4. 변수값 : "sk-"
*** 재부팅 또는 vscode나 cmd창 반드시 껐다가 킬 것
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.9,  # <- 여기로 이동
    messages=[
        {
        "role": "user",
        "content": "api 프로토콜에 대해 알려줘"
        }
    ]
)

# print(completion)
print("=================================")
print(completion.choices[0].message.content)
