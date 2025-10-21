from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

def get_ai_response(message):
    response = client.chat.completions.create(
        model = 'gpt-4o',
        temperature=0.9,
        messages=message,
    )
    return response.choices[0].message.content

messages = [
    {'role':'system', 'content':'너는 나를 서포트 하는 AI다.'},
]

while True:
    user_input = input("나: ")

    if user_input == 'exit':
        break

    messages.append({'role':'user','content':user_input})
    ai_response = get_ai_response(messages)
    messages.append({'role:':'assistant', 'content':ai_response})
    print('AI: ', ai_response)