from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

while True:
    user_input = input('나: ')

    if user_input == 'exit':
        break
    
    response = client.chat.completions.create(
        model = 'gpt-4o',
        temperature=0.9,
        messages = [
            {'role':'system', 'content':'너는 나를 서포트 하는 AI다.'},
            {'role':'user', 'content':user_input}
        ]
    )
    print('AI: ', response.choices[0].message.content)

    



