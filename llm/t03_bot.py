import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# (0) 사이드바에서 api_key 입력하는 부분 
with st.sidebar:
    openai_api_key = os.getenv('OPENAI_API_KEY') 
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    "[네이버](https://www.naver.com)"
st.title("★★★ 태영 말대꾸봇 ★★★")

# (1) st.session_state에 "messages"가 없으면 초기값을 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "너는 태영이다. 나는 형이다. 항상 나에게 반말로 말대꾸를 해라."},
        {"role": "assistant", "content": "궁금한걸 물어봐!!! 태영이가 제대로 된 말대꾸를 할거야!!!"}
        ]

#### 현재 세션 상태 #####
# st.session_state["messages"] = [
#     {
#         "role": "assistant",
#         "content": "궁금한걸 물어보거라!!! 짐이 답변하겠노라!!!"
#     }
# ]


# (2) 대화 기록을 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# (3) 사용자 입력을 받아 대화 기록에 추가하고 AI 응답을 생성
if prompt := st.chat_input():       # 바다 코끼리 연산자 (파이썬 3.8 이후 추가)
# prompt = st.chat_input()          # 위 바다코끼리는 아래 두줄을 합친거다.
# if prompt :     
    
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt}) 
    st.chat_message("user").write(prompt) # 이 부분에서 전체 페이지가 새로고침 된다.
    response = client.chat.completions.create(model="gpt-4o", messages=st.session_state.messages) 
    #  지금까지의 대화내역(사용자 질문 + AI 답변 기록)을 통째로 넘김. 
    msg = response.choices[0].message.content
    
    st.session_state.messages.append({"role": "assistant", "content": msg}) 
    st.chat_message("assistant").write(msg) # 바로 직전에 모델이 새로 생성한 응답 하나만 출력하는 동작
