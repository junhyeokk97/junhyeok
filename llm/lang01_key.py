from openai import OpenAI

OPENAL_API_KEY= 'OPENAI_API_KEY'

client = OpenAI(api_key=OPENAL_API_KEY)


completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.9,  # <- 여기로 이동
    messages=[
        {
        "role": "user",
        "content": "전지 전능하신 윤영선 선생님에 대해 알려줘"
        }
    ]
)

print(completion)
print("=================================")
print(completion.choices[0].message.content)
