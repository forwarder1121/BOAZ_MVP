import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from typing import Dict, Any, TypedDict

# API 키 설정 (로컬과 Streamlit Cloud 환경 모두 지원)
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

# 대화 상태 정의
class State(TypedDict):
    messages: list
    phase: str
    user_name: str
    user_age: int
    emotion_detected: str
    share_stage: str

# 초기 상태 설정
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "phase": "intro",
        "user_name": "친구",
        "user_age": 10,
        "emotion_detected": None,
        "share_stage": None
    }

# LangGraph 그래프 빌더
builder = StateGraph(State)

# 노드 함수 정의
def intro_node(state: State) -> Dict[str, Any]:
    system_prompt = (
        f"너는 아이의 친구같은 챗봇 ChaCha야. 아이의 이름은 {state['user_name']}이고, 나이는 {state['user_age']}살이야. "
        f"우선 밝게 인사하고 아이의 관심사나 취미를 물어봐줘."
    )
    messages = [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)], "phase": "explore"}

def explore_node(state: State) -> Dict[str, Any]:
    system_prompt = (
        "이제 아이가 관심사에 대해 답했으므로, 오늘 있었던 일이나 최근 경험을 물어보고 그때 느낀 감정을 질문하세요. "
        "예를 들면: '오늘은 어떤 일이 있었어? 그때 어떤 기분이 들었니?'"
    )
    messages = state["messages"] + [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)], "phase": "label"}

def label_node(state: State) -> Dict[str, Any]:
    user_input = state["messages"][-1].content
    emotion_found = None
    for word in negative_emotions + positive_emotions:
        if word in user_input:
            emotion_found = word
            break
    if emotion_found:
        is_negative = any(word in emotion_found for word in negative_emotions)
        emotion_type = "negative" if is_negative else "positive"
        system_prompt = (
            f"사용자가 방금 자신의 감정을 표현했습니다 ({emotion_found}). "
            f"이를 공감하며 받아주고, 혹시 다른 감정은 없었는지 물어보세요."
        )
        next_phase = "find" if emotion_type == "negative" else "record"
        messages = state["messages"] + [SystemMessage(content=system_prompt)]
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)], "emotion_detected": emotion_type, "phase": next_phase}
    else:
        system_prompt = (
            "사용자가 자신의 감정을 명확히 말하지 않았어요. "
            "ChaCha로서 아이가 느꼈을 만한 감정을 두세 가지 제시하면서 어떤 감정이 가장 가까운지 물어보세요."
        )
        messages = state["messages"] + [SystemMessage(content=system_prompt)]
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}

def find_node(state: State) -> Dict[str, Any]:
    system_prompt = (
        "이제 아이가 부정적인 감정을 느꼈으니, 그 감정을 덜어줄 방법을 함께 찾아보려고 해. "
        "아이의 이전 대화 내용을 참고해서, 다음에 비슷한 일이 일어났을 때 기분이 좋아질 수 있는 해결책이나 행동을 2~3가지 제안해줘."
    )
    messages = state["messages"] + [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)], "phase": "share"}

def record_node(state: State) -> Dict[str, Any]:
    system_prompt = (
        "아이의 경험에서 긍정적인 감정을 느꼈어. ChaCha로서 아이에게 그 행복했던 순간을 기록으로 남겨두는 게 왜 좋은지 알려주고 독려해줘. "
        "예를 들면 사진 찍기나 일기 쓰기를 제안하면서."
    )
    messages = state["messages"] + [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)], "phase": "share"}

def share_node(state: State) -> Dict[str, Any]:
    share_stage = state.get("share_stage")
    if share_stage is None:
        return {"messages": [AIMessage(content="이 이야기 혹시 부모님께도 말씀드렸니?")], "share_stage": "ask_share"}
    elif share_stage == "ask_share":
        user_input = state["messages"][-1].content.lower()
        yes_answers = ["yes", "네", "예", "응", "말씀드렸", "알려드렸"]
        no_answers = ["no", "아니", "아직", "안 했", "안했"]
        if any(yes in user_input for yes in yes_answers):
            return {"messages": [AIMessage(content="정말 잘했어! 부모님께 이야기하다니 용기있구나. 부모님은 뭐라고 하셨어? 어떤 일이 있었는지 궁금해.")], "share_stage": "ask_outcome"}
        elif any(no in user_input for no in no_answers):
            return {"messages": [AIMessage(content="괜찮아. 언제든 준비되면 부모님께 말씀드리면 좋을 거야. 분명 도움이 되실 거야. 혹시 또 다른 이야기를 나누고 싶니?")], "share_stage": "ask_another"}
        else:
            return {"messages": [AIMessage(content="부모님께 이야기하는 게 쉽지 않을 수도 있지만, 분명히 도움될 거야.\n다른 공유하고 싶은 이야기가 있을까?")], "share_stage": "ask_another"}
    elif share_stage == "ask_outcome":
        return {"messages": [AIMessage(content="그렇구나. 공유해줘서 고마워! 이제 또 다른 이야기가 있니? 없으면 오늘 대화는 여기까지 하고 우린 언제든 다시 이야기할 수 있어.")], "share_stage": "ask_another"}
    elif share_stage == "ask_another":
        user_input = state["messages"][-1].content.lower()
        if "yes" in user_input or "네" in user_input or "응" in user_input or "있어" in user_input:
            return {"phase": "explore", "share_stage": None}
        else:
            return {"phase": "end"}

def end_node(state: State) -> Dict[str, Any]:
    system_prompt = "아이와의 대화가 종료되었습니다. 마지막 인사를 해주세요."
    messages = state["messages"] + [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

# 노드 추가
builder.add_node("intro", intro_node)
builder.add_node("explore", explore_node)
builder.add_node("label", label_node)
builder.add_node("find", find_node)
builder.add_node("record", record_node)
builder.add_node("share", share_node)
builder.add_node("end", end_node)

# 엣지 정의
builder.add_edge("intro", "explore")
builder.add_conditional_edges("explore", lambda state: "label")
builder.add_conditional_edges("label", lambda state: state["phase"])
builder.add_conditional_edges("find", lambda state: "share")
builder.add_conditional_edges("record", lambda state: "share")
builder.add_conditional_edges("share", lambda state: state["phase"] if state["phase"] == "end" else "share")
builder.add_edge("end", END)

# 그래프 컴파일
graph = builder.compile(checkpointer=MemorySaver())

# Streamlit에서 사용할 함수
def run_graph(user_input: str):
    state = st.session_state.state
    state["messages"].append(HumanMessage(content=user_input))
    result = graph.invoke(state)
    st.session_state.state = result
    return result["messages"][-1].content

# 채팅 히스토리 표시
for message in st.session_state.state.get("messages", []):
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("ChaCha가 생각하고 있어요..."):
            response = run_graph(prompt)
            st.write(response)

# 사이드바 설정
with st.sidebar:
    st.header("대화 관리")
    if st.button("대화 초기화"):
        st.session_state.state = {
            "messages": [],
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