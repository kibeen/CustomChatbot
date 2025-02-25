import os
import streamlit as st
from utils import *
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


MODEL = "deepseek-r1:14b"

# Streamlit 페이지 설정
st.set_page_config(page_title="ChatGB")
st.title("💬 ChatGB")
st.write(f"Model: {MODEL}")


def init_prompt():
    # **한국어로 답변하도록 기본 프롬프트 설정**
    system_prompt = """
    당신은 한국어를 사용하는 AI 챗봇입니다.
    모든 질문에 대해 한국어로 답변하세요.
    문장이 자연스럽고 이해하기 쉬운 방식으로 답변해 주세요.
    대화의 문맥을 이해하고, 앞선 대화 내용과 연관된 답변을 제공하세요.
    """

    return PromptTemplate(
        template=system_prompt + "\n대화 내역: {history}\n사용자 질문: {input}\nAI 응답:",
        input_variables=["history", "input"]
    )


# Ollama DeepSeek 모델을 사용하여 챗봇 생성
@st.cache_resource
def create_ollama_chatbot():
    from langchain_community.llms import Ollama
    from langchain.chains import LLMChain
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    chatbot = LLMChain(
        llm=Ollama(model=MODEL),
        memory=memory,
        prompt=init_prompt()
    )
    return chatbot

# Ollama DeepSeek 모델을 사용하여 챗봇 생성
@st.cache_resource
def create_gpt_chatbot():
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationChain
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    os.environ["OPENAI_API_KEY"] = "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"
    chatbot = ConversationChain(
        llm=ChatOpenAI(model_name=MODEL),
        memory=memory
    )
    return chatbot

# Streamlit의 session_state에 챗봇 저장 (초기화 방지)
if "chatbot" not in st.session_state:
    with st.spinner("DeepSeek 챗봇 초기화 중입니다..."):
        chatbot = create_gpt_chatbot()
        st.session_state.chatbot = chatbot
    st.write("DeepSeek 챗봇이 준비되었습니다! 질문을 입력하세요.")

# 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 출력
for conversation in st.session_state.messages:
    with st.chat_message(conversation["role"]):
        st.write(conversation["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하면 챗봇이 답변을 제공합니다."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    raw_response = st.session_state.chatbot.run(prompt)  # 원본 응답 받기
    response = clean_response(raw_response)  # <think> 태그 제거 후처리
    
    # 응답 출력
    with st.chat_message("assistant"):
        st.markdown(response)

    # 대화 기록 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
