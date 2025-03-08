import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# API 키 설정
load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
api_key = os.environ.get("OPENAI_API_KEY")

# 감정 단어 리스트
negative_emotions = ["화나", "슬프", "속상", "우울", "불안", "걱정", "짜증", "힘들"]
positive_emotions = ["기쁘", "행복", "즐겁", "신나", "설레", "좋아", "재미"]

# Streamlit 페이지 설정
st.set_page_config(page_title="ChaCha - 아이들을 위한 감정 대화 챗봇", page_icon="🤖")
st.title("ChaCha와 대화하기 🤖")

# OpenAI API 초기화
try:
    if not api_key:
        st.warning("OpenAI API 키가 설정되지 않았습니다. 환경 변수나 secrets를 확인해주세요.")
        llm = None
    else:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
except Exception as e:
    st.error(f"API 초기화 오류: {str(e)}")
    llm = None

# 대화 상태 관리
if "messages" not in st.session_state:
    st.session_state.messages = []
if "state" not in st.session_state:
    st.session_state.state = {
        "phase": "intro",
        "user_name": "친구",
        "user_age": 10,
        "emotion_detected": None,
        "share_stage": None
    }

# 채팅 처리 함수 (이전과 동일, 생략)
def process_message(user_input: str):
    try:
        current_phase = st.session_state.state["phase"]
        user_name = st.session_state.state["user_name"]
        user_age = st.session_state.state["user_age"]

        if current_phase == "intro":
            system_prompt = (
                f"너는 아이의 친구같은 챗봇 ChaCha야. 아이의 이름은 {user_name}이고, 나이는 {user_age}살이야. "
                f"우선 밝게 인사하고 아이의 관심사나 취미를 물어봐줘."
            )
            st.session_state.state["phase"] = "explore"
        # 나머지 로직은 동일

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        history = st.session_state.messages[-10:] if len(st.session_state.messages) > 0 else []
        if history:
            messages = history + messages

        if llm is None:
            return "API 설정 오류로 응답할 수 없습니다. API 키를 확인해주세요."
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        st.error(f"메시지 처리 중 오류가 발생했습니다: {str(e)}")
        return "죄송해요, 대화 처리 중 문제가 발생했어요. 다시 시도해 주세요."

# 채팅 히스토리 및 입력 처리 (이전과 동일, 생략)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

if prompt := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("ChaCha가 생각하고 있어요..."):
            response = process_message(prompt)
            st.write(response)
            st.session_state.messages.append(AIMessage(content=response))

# 사이드바 (이전과 동일, 생략)
with st.sidebar:
    st.header("대화 관리")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.state = {
            "phase": "intro",
            "user_name": "친구",
            "user_age": 10,
            "emotion_detected": None,
            "share_stage": None
        }
        st.rerun()

    st.header("사용자 정보")
    new_name = st.text_input("이름", value=st.session_state.state["user_name"])
    new_age = st.number_input("나이", min_value=1, max_value=20, value=st.session_state.state["user_age"])
    if new_name != st.session_state.state["user_name"] or new_age != st.session_state.state["user_age"]:
        st.session_state.state["user_name"] = new_name
        st.session_state.state["user_age"] = new_age