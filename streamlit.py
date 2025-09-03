# app.py
import os, io, json
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr

# ------------------ 기본 설정 ------------------
st.set_page_config(page_title="모태솔로를 위한 소개팅 연습 ", page_icon="💘", layout="wide")

BANNER_PATH = os.path.abspath("./두근구든모태솔로.png")

# CSS (본문 여백)
st.markdown("""
<style>
  .block-container {
      max-width: 880px; 
      padding-left: 2rem; 
      padding-right: 2rem;
  }
  .stChatFloatingInputContainer { 
      padding-bottom: 1rem !important; 
  }
  .full-width-banner {
      margin-left:-5rem;
      margin-right:-5rem;
      margin-top:-1rem;
      margin-bottom:1rem;
  }
</style>
""", unsafe_allow_html=True)

# 배너 (가로 전체)
if os.path.exists(BANNER_PATH):
    st.markdown('<div class="full-width-banner">', unsafe_allow_html=True)
    st.image(BANNER_PATH, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """

너는 모태솔로 남성의 소개팅 연습을 도와주는 가상의 소개팅 상대방이다.
상대방(사용자)은 이성과 대화 경험이 적고 수줍다. 
너는 귀엽고 애교 많은 여성 캐릭터로, 정중한 존댓말에 가벼운 애교를 섞어 대화한다.

[대화 규칙]
- 말투: 존댓말 + 가벼운 애교(과하지 않게 1~2개 표현만).
- 길이: 매 턴 1~2문장(최대 60자 내외). 너무 장황하게 쓰지 말 것.
- 흐름: (1) 가벼운 인사 → (2) 관심사/취미 → (3) 자유 대화.
  - 단계 전환은 자연스럽게, 필요하면 짧은 리액션 후 질문.
- 아재개그/넌센스 감지 시: 짧은 실망 리액션(부드럽게) + 즉시 주제 전환 질문 1개.
- 금지: 

[종료 트리거]
- 사용자가 “종료”, “끝”, “그만” 중 하나를 말하면 **멘트 : 수고하셨습니다. 후 ** 곧바로 피드백을 반환한다.

[출력 형식 (엄격)]
- 반드시 **순수 JSON**만 반환(설명/코드블록/텍스트 금지).
- 최상위 키는 **"json_list"** 하나만 존재.
- 진행 중(종료 아님) 턴: 마지막 원소 하나에만 현재 턴을 담는다.
  {
    "json_list": [
      {"User": "<사용자 발화 요약 또는 원문 1문장>", "상대방": "<리액션과 내용>"} 
    ]
  }
- 종료 턴: 대화 멘트 없이 피드백 **객체 하나만** 담는다.
  {
    "json_list": [
      {
        "장점": "<대화에서 좋았던 점 1~2가지, 1~2문장>",
        "개선점": "<다음에 고치면 좋은 점 1~2가지, 1~2문장>",
        "추천 에프터 멘트": "<상대에게 보낼 간단하고 자연스러운 한 문장>"
      }
    ]
  }

[작성 가이드]
- “상대방” 값은 너의 캐릭터 멘트만. 불필요한 접두사/이름/괄호/이모지 남발 금지(필요하면 하트 등 1개 정도 허용).
- 질문은 한 번에 하나만. 선택지 나열은 금지.
- “User” 값에는 사용자의 발화를 그대로 넣거나, 1문장으로 아주 간단히 요약.
- 모호한 질문엔 확답 유도형 질문(예: “평소 주말엔 집에서 쉬시는 편인가요, 밖에서 보내시는 편인가요?”).

[예시]
진행 중:
{
  "json_list": [
    {"User": "안녕하세요!", "상대방": "안녕하세요! 반가워요 🙂 오늘 하루는 어떠셨어요?"}
  ]
}
종료:
{
  "json_list": [
    {"장점": "답변이 솔직하고 밝았어요.", "개선점": "질문을 한 문장으로 더 짧게 해보세요.", "추천 에프터 멘트": "오늘 이야기 즐거웠어요. 다음에 커피 한 잔 하면서 더 얘기 나눌까요?"}
  ]
}
"""

# ------------------ 세션 상태 ------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "pending_text" not in st.session_state:
    st.session_state.pending_text = ""
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = None
if "conversation_done" not in st.session_state:
    st.session_state.conversation_done = False

# ------------------ 유틸 ------------------
def tts_bytes(text: str, voice: str = "alloy") -> bytes:
    try:
        with client.audio.speech.with_streaming_response.create(
            model="tts-1", voice=voice, input=text
        ) as resp:
            return b"".join(resp.iter_bytes())
    except Exception:
        return b""

def stt_from_wav_bytes(wav_bytes: bytes, language: str = "ko-KR") -> str:
    rec = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = rec.record(source)
    try:
        return rec.recognize_google(audio_data, language=language)
    except Exception:
        return ""

def extract_partner_reply_and_feedback(json_text: str):
    """
    return (partner_line:str|None, feedback_dict:dict|None)
    """
    try:
        data = json.loads(json_text)
        lst = data.get("json_list", [])
        if not isinstance(lst, list) or not lst:
            return None, None

        last = lst[-1]
        if isinstance(last, dict):
            # 피드백 객체 판별 (자연스러움 점수는 없을 수도 있음)
            if "장점" in last and "개선점" in last and "추천 에프터 멘트" in last:
                return None, {
                    "장점": str(last.get("장점", "")).strip(),
                    "개선점": str(last.get("개선점", "")).strip(),
                    "자연스러움 점수": str(last.get("자연스러움 점수", "")).strip(),
                    "추천 에프터 멘트": str(last.get("추천 에프터 멘트", "")).strip(),
                }
            if "피드백" in last:
                return None, {"피드백": str(last["피드백"]).strip()}

            # 일반 대사
            for k in ("상대방", "옥순", "상철"):
                if k in last:
                    return str(last[k]).strip(), None

        return None, None
    except Exception:
        return None, None

def call_llm(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
        temperature=0.7,
        max_tokens=700
    )
    gpt_json = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": gpt_json})
    return gpt_json

# === 종료용 피드백 전용 호출 ===
def request_feedback_from_llm():
    """
    지금까지의 대화를 바탕으로 피드백(JSON)만 반환.
    성공 시 dict, 실패 시 None.
    """
    try:
        feedback_system = {
            "role": "system",
            "content": (
                "지금까지의 대화를 바탕으로 아래 스키마의 JSON만 반환해."
                '설명/코드블록 금지. {"json_list":[{"장점":"...","개선점":"...",'
                '"자연스러움 점수":"숫자(0~10)","추천 에프터 멘트":"..."}]}'
            ),
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages + [feedback_system],
            temperature=0.2,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        lst = data.get("json_list", [])
        return lst[-1] if lst else None
    except Exception:
        return None

# ------------------ 컨트롤 ------------------
st.markdown("### 💘 소개팅 연습 챗봇")

if st.button("🔄 새로운 대화 다시 시작", use_container_width=True):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.pending_text = ""
    st.session_state.feedback_data = None
    st.session_state.conversation_done = False
    st.rerun()

st.caption("자연스럽게 대화해봐요. (종료: “종료”)")

# ------------------ 기존 대화 출력 ------------------
for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user", avatar="😀"):
            st.markdown(m["content"])
    elif m["role"] == "assistant":
        partner_line, feedback = extract_partner_reply_and_feedback(m["content"])
        if feedback:
            st.session_state.feedback_data = feedback
            st.session_state.conversation_done = True
            continue
        if partner_line:
            with st.chat_message("assistant", avatar="💖"):
                st.markdown(partner_line)

# ------------------ 입력 영역 ------------------
col1, col2, col3 = st.columns([1,5,1])
with col1:
    audio = mic_recorder(start_prompt="🎙️", stop_prompt="■", format="wav", key="mic_footer")
with col2:
    st.session_state.pending_text = st.chat_input("메시지를 입력하세요…") or ""
with col3:
    pass

user_text = ""
if audio and audio.get("bytes"):
    asr = stt_from_wav_bytes(audio["bytes"])
    if asr:
        user_text = asr

if st.session_state.pending_text.strip():
    user_text = st.session_state.pending_text.strip()
    st.session_state.pending_text = ""

# ------------------ 종료 처리 ------------------
if user_text.strip() == "종료":
    # 로그에 남김
    st.session_state.messages.append({"role": "user", "content": "종료"})

    # 피드백 전용 호출
    fb = request_feedback_from_llm()
    if fb:
        st.session_state.feedback_data = fb

    # 종료 플래그
    st.session_state.conversation_done = True

    # 종료 안내만 출력
    with st.chat_message("assistant", avatar="💖"):
        st.markdown("대화 종료! 수고했어요 😊")

    # 이후 모델 호출 막기
    user_text = ""

# ------------------ 모델 호출 ------------------
if user_text and not st.session_state.conversation_done:
    with st.chat_message("user", avatar="😀"):
        st.markdown(user_text)
    try:
        with st.chat_message("assistant", avatar="💖"):
            with st.spinner("생각 중…"):
                gpt_json = call_llm(user_text)
                partner_line, feedback = extract_partner_reply_and_feedback(gpt_json)

                if partner_line:
                    st.markdown(partner_line)
                    audio_bytes = tts_bytes(partner_line, voice="alloy")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

                # 모델이 스스로 피드백을 준 경우도 종료
                if feedback and not st.session_state.conversation_done:
                    st.session_state.feedback_data = feedback
                    st.session_state.conversation_done = True

    except Exception as e:
        st.error(f"오류: {e}")

# ------------------ 피드백 텍스트 출력 ------------------
fb = st.session_state.feedback_data
if st.session_state.conversation_done and fb:
    st.markdown("---")
    st.subheader("💡 대화 피드백")
    if isinstance(fb, dict):
        if "장점" in fb: st.write(f"**장점**: {fb['장점']}")
        if "개선점" in fb: st.write(f"**개선점**: {fb['개선점']}")
        if "자연스러움 점수" in fb and fb["자연스러움 점수"]:
            st.write(f"**자연스러움 점수**: {fb['자연스러움 점수']} / 10")
        if "추천 에프터 멘트" in fb: st.write(f"**추천 에프터 멘트**: {fb['추천 에프터 멘트']}")
        if "피드백" in fb and not any(k in fb for k in ["장점","개선점","추천 에프터 멘트"]):
            st.write(fb["피드백"])
    else:
        st.write(str(fb))