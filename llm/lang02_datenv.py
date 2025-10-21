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
