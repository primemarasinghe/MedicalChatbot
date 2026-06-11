import streamlit as st
import os
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🩺",
    layout="wide",
)

# ── Helper ────────────────────────────────────────────────────────────────────
def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BG_PATH  = "static/medback.png"
BOT_PATH = "static/bot.png"

bg_b64  = img_to_base64(BG_PATH)  if os.path.exists(BG_PATH)  else None
bot_b64 = img_to_base64(BOT_PATH) if os.path.exists(BOT_PATH) else None

bot_avatar = BOT_PATH if os.path.exists(BOT_PATH) else "🩺"

bg_css = f"""
  .stApp {{
    background-image: url("data:image/png;base64,{bg_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
  }}
  .stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(4, 7, 14, 0.78);
    z-index: 0;
  }}
""" if bg_b64 else ".stApp { background: #05080f; }"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#MainMenu, footer, header {{ visibility: hidden; }}

* {{ font-family: 'Inter', sans-serif !important; }}

{bg_css}

/* All content above overlay */
.stApp > * {{ position: relative; z-index: 1; }}
[data-testid="stAppViewContainer"] > section {{ position: relative; z-index: 1; }}
[data-testid="stMain"] {{ position: relative; z-index: 1; }}

/* ── Layout: constrain width + center ── */
.block-container {{
  max-width: 860px !important;
  margin: 0 auto !important;
  padding: 1.5rem 1.5rem 6rem 1.5rem !important;
}}

/* ── Chat messages — much bigger & animated ── */
[data-testid="stChatMessage"] {{
  background: rgba(10, 18, 34, 0.88) !important;
  border: 1px solid rgba(30, 55, 90, 0.9) !important;
  border-radius: 18px !important;
  padding: 20px 24px !important;
  margin-bottom: 14px !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
  animation: msgSlide 0.35s cubic-bezier(.22,.68,0,1.2) forwards;
  font-size: 15px !important;
  line-height: 1.75 !important;
}}

@keyframes msgSlide {{
  from {{ opacity: 0; transform: translateY(14px) scale(0.98); }}
  to   {{ opacity: 1; transform: translateY(0)   scale(1);    }}
}}

/* Avatar bigger */
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarContainer"] {{
  width: 44px !important;
  height: 44px !important;
  border-radius: 50% !important;
  border: 2px solid #0e4a6e !important;
  overflow: hidden !important;
}}

/* Message text */
[data-testid="stChatMessage"] p {{
  color: #d4e4ff !important;
  font-size: 15px !important;
  line-height: 1.75 !important;
  margin: 0 !important;
}}

/* ── Input bar ── */
[data-testid="stBottom"] {{
  background: rgba(6, 10, 20, 0.92) !important;
  backdrop-filter: blur(14px) !important;
  border-top: 1px solid rgba(26, 42, 66, 0.8) !important;
  padding: 14px 0 !important;
}}

[data-testid="stChatInputTextArea"] {{
  background: rgba(13, 22, 42, 0.95) !important;
  color: #c8d8f0 !important;
  border: 1.5px solid #1a3a5c !important;
  border-radius: 28px !important;
  font-size: 14px !important;
  padding: 14px 20px !important;
  transition: border-color 0.2s !important;
}}

[data-testid="stChatInputTextArea"]:focus {{
  border-color: #1a9fd4 !important;
  box-shadow: 0 0 0 3px rgba(26,159,212,0.12) !important;
}}

/* ── Quick chip buttons ── */
.stButton > button {{
  background: rgba(10, 18, 34, 0.82) !important;
  border: 1px solid rgba(30, 55, 90, 0.8) !important;
  color: #5a9fd4 !important;
  border-radius: 22px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 8px 4px !important;
  backdrop-filter: blur(6px) !important;
  transition: all 0.18s ease !important;
  letter-spacing: 0.01em !important;
}}

.stButton > button:hover {{
  background: rgba(14, 31, 60, 0.95) !important;
  color: #1a9fd4 !important;
  border-color: #0e4a6e !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(26,159,212,0.18) !important;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] {{
  color: #1a9fd4 !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar       {{ width: 4px; }}
::-webkit-scrollbar-thumb {{ background: #1a3050; border-radius: 4px; }}

/* Text */
h1, h2, h3 {{ color: #e8f0fe !important; }}
p, span {{ color: #c8d8f0; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
bot_img_tag = (
    f'<img src="data:image/png;base64,{bot_b64}" '
    f'style="width:54px;height:54px;border-radius:50%;object-fit:cover;'
    f'border:2px solid #1a9fd4;" />'
    if bot_b64
    else '<div style="width:54px;height:54px;border-radius:50%;background:#0d2a3d;'
         'border:2px solid #1a9fd4;display:flex;align-items:center;'
         'justify-content:center;font-size:26px;">🩺</div>'
)

st.markdown(f"""
<div style="
  display:flex; align-items:center; gap:16px;
  padding: 18px 0 10px 0;
  border-bottom: 1px solid rgba(26,42,66,0.6);
  margin-bottom: 22px;
  animation: fadeDown 0.5s ease forwards;
">
  <!-- Bot avatar with pulse ring -->
  <div style="position:relative;flex-shrink:0;">
    {bot_img_tag}
    <div style="
      position:absolute; inset:-5px; border-radius:50%;
      border:1.5px solid #1a9fd4;
      animation:ringpulse 2.4s ease-out infinite;
    "></div>
  </div>

  <!-- Title block -->
  <div style="flex:1;">
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <span style="font-size:22px;font-weight:700;color:#e8f0fe;letter-spacing:-0.01em;">
        MediAssist AI
      </span>
      <span style="font-size:10px;padding:3px 10px;border-radius:20px;
                   background:rgba(2,31,24,0.85);color:#00d97e;
                   border:1px solid #004d30;font-weight:600;">🛡 Secure</span>
      <span style="font-size:10px;padding:3px 10px;border-radius:20px;
                   background:rgba(2,24,48,0.85);color:#1a9fd4;
                   border:1px solid #0e4a6e;font-weight:600;">v2.4</span>
    </div>
    <div style="font-size:12px;color:#3a7ca5;margin-top:5px;display:flex;align-items:center;gap:6px;">
      <span style="width:7px;height:7px;background:#00d97e;border-radius:50%;
                   display:inline-block;animation:blink 2s infinite;"></span>
      Online &nbsp;·&nbsp; RAG System powered by Llama&nbsp;3.3&nbsp;70B + Pinecone
    </div>
    <div style="font-size:11px;color:#ffffff;margin-top:4px;font-weight:500;">
      Built by&nbsp;
      <span style="color:#1a9fd4;font-weight:700;">Primesh Marasinghe</span>
      &nbsp;·&nbsp; Retrieval-Augmented Generation Medical Assistant
    </div>
  </div>

  <!-- RAG badge -->
  <div style="flex-shrink:0;text-align:center;">
    <div style="font-size:10px;padding:5px 13px;border-radius:20px;
                background:rgba(30,10,60,0.85);color:#a78bfa;
                border:1px solid #3b1f80;font-weight:600;letter-spacing:0.04em;">
      ⚡ RAG System
    </div>
  </div>
</div>

<style>
@keyframes fadeDown {{
  from {{ opacity:0; transform:translateY(-10px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes ringpulse {{
  0%   {{ opacity:0.8; transform:scale(1);   }}
  100% {{ opacity:0;   transform:scale(1.35);}}
}}
@keyframes blink {{
  0%,100% {{ opacity:1;    }}
  50%      {{ opacity:0.2; }}
}}
</style>
""", unsafe_allow_html=True)

# ── Load env & RAG chain ──────────────────────────────────────────────────────
load_dotenv()

@st.cache_resource(show_spinner="⚙️ Initialising RAG pipeline…")
def build_rag_chain():
    for key in ("PINECONE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        os.environ[key] = os.getenv(key, "")

    embedding = download_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embedding,
        index_name="medicalbot",
    )
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

rag_chain = build_rag_chain()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm **MediAssist AI**, your intelligent medical assistant "
                "built on a RAG (Retrieval-Augmented Generation) system.\n\n"
                "I can help you with **symptoms**, **medications**, **vaccines**, "
                "**first aid**, and general health guidance. "
                "How are you feeling today?"
            ),
        }
    ]

# ── Quick chips ───────────────────────────────────────────────────────────────
CHIPS = [
    ("💊", "Medication info"),
    ("🩺", "Check symptoms"),
    ("💉", "Vaccines"),
    ("🩹", "First aid"),
    ("🧬", "Drug interactions"),
    ("❤️", "Heart health"),
]

if not st.session_state.get("chips_hidden"):
    cols = st.columns(len(CHIPS))
    for col, (icon, label) in zip(cols, CHIPS):
        with col:
            if st.button(f"{icon} {label}", use_container_width=True, key=label):
                st.session_state.pending_chip = label
                st.session_state.chips_hidden = True

    st.markdown(
        '<div style="text-align:center;font-size:11px;color:#1e3a56;'
        'margin-top:4px;margin-bottom:8px;">Quick questions — tap to ask</div>',
        unsafe_allow_html=True,
    )

# ── Chat bubble HTML helpers ──────────────────────────────────────────────────
bot_av_html = (
    f'<img src="data:image/png;base64,{bot_b64}" '
    f'style="width:40px;height:40px;border-radius:50%;object-fit:cover;'
    f'border:2px solid #1a9fd4;flex-shrink:0;" />'
    if bot_b64
    else '<div style="width:40px;height:40px;border-radius:50%;background:#0d2a3d;'
         'border:2px solid #1a9fd4;display:flex;align-items:center;'
         'justify-content:center;font-size:20px;flex-shrink:0;">🩺</div>'
)

user_av_html = (
    '<div style="width:40px;height:40px;border-radius:50%;background:#1a0d38;'
    'border:2px solid #6d4aff;display:flex;align-items:center;'
    'justify-content:center;font-size:16px;font-weight:700;'
    'color:#a78bfa;flex-shrink:0;">You</div>'
)

import markdown as md_lib

def render_bubble(role: str, content: str) -> str:
    """Return full-width social-media style chat row HTML."""
    # Convert markdown bold/lists to HTML for display
    try:
        body = md_lib.markdown(content, extensions=["nl2br"])
    except Exception:
        body = content.replace("\n", "<br>")

    if role == "assistant":
        return f"""
<div style="display:flex;align-items:flex-start;gap:12px;
            margin-bottom:18px;animation:msgSlide 0.35s cubic-bezier(.22,.68,0,1.2) forwards;">
  {bot_av_html}
  <div style="max-width:78%;">
    <div style="font-size:11px;color:#3a7ca5;margin-bottom:5px;
                font-weight:600;letter-spacing:0.03em;">MediAssist AI</div>
    <div style="background:rgba(10,18,34,0.90);border:1px solid rgba(30,55,90,0.9);
                border-radius:4px 18px 18px 18px;padding:16px 20px;
                backdrop-filter:blur(12px);box-shadow:0 4px 24px rgba(0,0,0,0.3);
                font-size:15px;line-height:1.75;color:#d4e4ff;">
      {body}
    </div>
  </div>
</div>"""
    else:
        return f"""
<div style="display:flex;align-items:flex-start;gap:12px;
            flex-direction:row-reverse;
            margin-bottom:18px;animation:msgSlide 0.35s cubic-bezier(.22,.68,0,1.2) forwards;">
  {user_av_html}
  <div style="max-width:78%;display:flex;flex-direction:column;align-items:flex-end;">
    <div style="font-size:11px;color:#7c5cbf;margin-bottom:5px;
                font-weight:600;letter-spacing:0.03em;">You</div>
    <div style="background:rgba(20,10,48,0.90);border:1px solid rgba(80,40,160,0.6);
                border-radius:18px 4px 18px 18px;padding:16px 20px;
                backdrop-filter:blur(12px);box-shadow:0 4px 24px rgba(0,0,0,0.3);
                font-size:15px;line-height:1.75;color:#d4c4ff;">
      {body}
    </div>
  </div>
</div>"""

THINKING_HTML = """
<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:18px;" id="thinking-row">
  {bot_av}
  <div>
    <div style="font-size:11px;color:#3a7ca5;margin-bottom:5px;font-weight:600;">MediAssist AI</div>
    <div style="background:rgba(10,18,34,0.90);border:1px solid rgba(30,55,90,0.9);
                border-radius:4px 18px 18px 18px;padding:16px 20px;
                backdrop-filter:blur(12px);display:inline-flex;align-items:center;gap:12px;">
      <span style="font-size:13px;color:#5a9fd4;font-weight:500;
                   animation:textpulse 1.8s ease-in-out infinite;">🧠 AI is thinking</span>
      <div style="display:flex;align-items:flex-end;gap:5px;height:20px;">
        <div style="width:9px;height:9px;border-radius:50%;background:#1a9fd4;
                    animation:bounce 1.2s ease-in-out infinite;animation-delay:0s;"></div>
        <div style="width:9px;height:9px;border-radius:50%;background:#3ab8f0;
                    animation:bounce 1.2s ease-in-out infinite;animation-delay:0.2s;"></div>
        <div style="width:9px;height:9px;border-radius:50%;background:#6dd4ff;
                    animation:bounce 1.2s ease-in-out infinite;animation-delay:0.4s;"></div>
      </div>
    </div>
  </div>
</div>
<style>
@keyframes textpulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.45}} }}
@keyframes bounce    {{ 0%,80%,100%{{transform:translateY(0);opacity:0.6}} 40%{{transform:translateY(-10px);opacity:1}} }}
</style>
"""

# ── Chat history ──────────────────────────────────────────────────────────────
# Hide default st.chat_message styling since we use raw HTML now
st.markdown("""
<style>
/* Remove all default Streamlit chat chrome */
[data-testid="stChatMessage"]          { display:none !important; }
[data-testid="stChatMessageContainer"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# Render all past messages as custom HTML
history_html = "".join(render_bubble(m["role"], m["content"]) for m in st.session_state.messages)
st.markdown(history_html, unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Describe your symptoms or ask a health question…")

if "pending_chip" in st.session_state:
    user_input = st.session_state.pop("pending_chip")

# ── Response ──────────────────────────────────────────────────────────────────
if user_input:
    st.session_state.chips_hidden = True
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show user bubble immediately
    st.markdown(render_bubble("user", user_input), unsafe_allow_html=True)

    # Show thinking animation
    thinking_slot = st.empty()
    thinking_slot.markdown(
        THINKING_HTML.replace("{bot_av}", bot_av_html),
        unsafe_allow_html=True,
    )

    # RAG call
    response = rag_chain.invoke({"input": user_input})
    answer   = response["answer"]

    # Replace thinking with real answer
    thinking_slot.empty()
    st.markdown(render_bubble("assistant", answer), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:11px;color:#1a2a40;
            margin-top:30px;padding-top:14px;
            border-top:1px solid rgba(26,42,66,0.3);">
  MediAssist AI &nbsp;·&nbsp; RAG System &nbsp;·&nbsp;
  Built by <span style="color:#1a9fd4;">Primesh Marasinghe</span>
  &nbsp;·&nbsp; Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)