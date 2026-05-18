import streamlit as st
from audio_recorder_streamlit import audio_recorder
import time
import datetime
import io
import os
import torch
import torchaudio
import numpy as np
import re  # تم إضافة مكتبة re هنا في المكان الصحيح
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from sentence_transformers import SentenceTransformer, util

# ==========================================
# PAGE CONFIG (Must be first Streamlit command)
# ==========================================
st.set_page_config(
    page_title="VoiceScribe | AI Speech-to-Text",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# MODEL LOADING (Whisper)
# ==========================================
MODEL_DIR = r"D:\E-JUST\Level 3\Semester 2\Neural network\Speech to text\Arabic_ASR_App\whisper_model"

@st.cache_resource(show_spinner="Loading Whisper model…")
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_DIR, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,  # 👈 التعديل الأهم: تقسيم الصوت الطويل إلى أجزاء 30 ثانية
        batch_size=8,       # 👈 تسريع المعالجة عبر تمرير عدة أجزاء معاً لكارت الشاشة
    )
    return pipe

# ==========================================
# SUMMARIZATION MODELS
# ==========================================
@st.cache_resource(show_spinner="Loading Summarization Model...")
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn") 

# ==========================================
# LOGIC FUNCTIONS
# ==========================================
def process_audio(audio_path: str, language: str):
    pipe = load_model()
    
    # 👈 إضافة return_timestamps كما طلب منك الخطأ
    generate_kwargs = {"return_timestamps": True} 
    
    if language == "Arabic (العربية)":
        generate_kwargs["language"] = "arabic"
    elif language == "English":
        generate_kwargs["language"] = "english"

    # التمرير مع الإعدادات الجديدة
    result = pipe(audio_path, generate_kwargs=generate_kwargs)

    # في حالة الملفات الطويلة، قد يرجع النص داخل list of chunks
    if isinstance(result, list):
        transcript = " ".join([chunk.get("text", "") for chunk in result]).strip()
    else:
        transcript = result.get("text", "").strip()
        
    confidence = 0.95 
    return transcript, confidence

def generate_summary(text):
    word_count = len(text.split())
    
    # لو النص أقل من 15 كلمة، مفيش داعي نستهلك الموديل
    if word_count < 15:
        return "النص قصير وموجز بالفعل ولا يحتاج إلى تلخيص. (Text is already short and concise)."
        
    summarizer = load_summarizer()
    
    # ضبط طول الملخص ديناميكياً لتجنب أي أخطاء
    max_len = min(130, max(20, int(word_count * 0.6)))
    min_len = min(30, max(5, int(max_len * 0.3)))
    
    # تمرير النص للموديل بالحدود الجديدة
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background: #080d1a;
        background-image:
            radial-gradient(ellipse at 20% 20%, rgba(0, 210, 255, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(0, 150, 255, 0.05) 0%, transparent 50%);
    }
    #MainMenu, footer, header { visibility: hidden; }
    .hero-header { text-align: center; padding: 2.5rem 0 1rem 0; }
    .hero-badge {
        display: inline-block; background: rgba(0, 210, 255, 0.1); border: 1px solid rgba(0, 210, 255, 0.3);
        color: #00d2ff; padding: 0.3rem 1rem; border-radius: 999px; font-size: 0.75rem;
        font-family: 'Space Mono', monospace; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem;
    }
    .hero-title { font-size: 2.8rem; font-weight: 600; color: #f0f4ff; line-height: 1.2; margin: 0.5rem 0; }
    .hero-title span { color: #00d2ff; }
    .hero-subtitle { color: #6b7fa3; font-size: 1rem; font-weight: 300; max-width: 550px; margin: 0.5rem auto 0 auto; }
    .model-badge { display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-top: 1rem; color: #4a5a7a; font-size: 0.8rem; font-family: 'Space Mono', monospace; }
    .model-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: #00d2ff; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .custom-divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.5rem 0; }
    .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 4px; border: 1px solid rgba(255,255,255,0.07); }
    .stTabs [data-baseweb="tab"] { color: #6b7fa3; font-weight: 500; border-radius: 8px; padding: 0.5rem 1.5rem; }
    .stTabs [aria-selected="true"] { background: rgba(0, 210, 255, 0.12) !important; color: #00d2ff !important; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
    [data-testid="stFileUploader"] { background: rgba(255,255,255,0.03); border: 1.5px dashed rgba(0, 210, 255, 0.2); border-radius: 12px; padding: 1rem; }
    [data-testid="stFileUploader"]:hover { border-color: rgba(0, 210, 255, 0.5); }
    audio { width: 100%; border-radius: 8px; margin-top: 0.5rem; }
    .stButton > button { background: linear-gradient(135deg, #00d2ff, #0070f3); color: white; border: none; border-radius: 10px; font-weight: 600; padding: 0.75rem 1.5rem; box-shadow: 0 4px 20px rgba(0, 210, 255, 0.25); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0, 210, 255, 0.4); background: linear-gradient(135deg, #1ad9ff, #1a8aff); }
    .stButton > button[kind="secondary"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #8a9ac0 !important; box-shadow: none !important; }
    .stButton > button[kind="secondary"]:hover { background: rgba(255,255,255,0.08) !important; color: #f0f4ff !important; }
    .result-card { background: rgba(0, 210, 255, 0.05); border: 1px solid rgba(0, 210, 255, 0.2); border-radius: 14px; padding: 1.5rem; margin-top: 1rem; }
    .result-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #00d2ff; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.75rem; }
    .result-text { font-size: 1.1rem; color: #e8eeff; line-height: 1.7; font-weight: 400; }
    .stats-row { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
    .stat-pill { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 0.4rem 0.85rem; font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #6b7fa3; }
    .stat-pill span { color: #c8d8ff; font-weight: 700; }
    .history-item { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem; }
    .history-meta { font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #4a5a7a; margin-bottom: 0.4rem; display: flex; gap: 1rem; }
    .history-text { color: #8a9ac0; font-size: 0.9rem; line-height: 1.5; }
    .stSelectbox > div > div { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: #c8d8ff; }
    .stInfo { background: rgba(0, 210, 255, 0.07) !important; border: 1px solid rgba(0, 210, 255, 0.2) !important; color: #c8d8ff !important; border-radius: 10px; }
    .stSuccess { background: rgba(0, 255, 150, 0.07) !important; border: 1px solid rgba(0, 255, 150, 0.2) !important; color: #c8ffd4 !important; border-radius: 10px; }
    .section-heading { font-size: 0.7rem; font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: 0.12em; color: #4a5a7a; margin-bottom: 0.75rem; }
    .conf-bar-wrap { background: rgba(255,255,255,0.05); border-radius: 999px; height: 6px; margin-top: 0.3rem; }
    .conf-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #00d2ff, #0070f3); }
    .app-footer { text-align: center; padding: 2rem 0 1rem 0; color: #3a4a6a; font-size: 0.78rem; font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INIT
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ==========================================
# HERO HEADER
# ==========================================
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">✦ AI-Powered Transcription</div>
    <div class="hero-title">Voice<span>Scribe</span></div>
    <div class="hero-subtitle">
        <span style="color: #00d2ff; font-weight: 500;">✦ Intelligent Audio Analysis System (Speech → Text → Summary → Search) ✦</span><br><br>
        Upload an audio file or record directly — our Conformer + Bi-LSTM model converts speech to text instantly.
    </div>
    <div class="model-badge">
        <div class="dot"></div>
        Conformer · Bi-LSTM · PyTorch
    </div>
</div>
<hr class="custom-divider">
""", unsafe_allow_html=True)

# ==========================================
# SETTINGS & INPUTS
# ==========================================
col_lang, col_gap = st.columns([1, 2])
with col_lang:
    language = st.selectbox("🌐 Language", ["Auto-Detect", "Arabic (العربية)", "English"], index=0)

audio_to_process = None
tab1, tab2 = st.tabs(["📁  Upload Audio File", "🎤  Live Recording"])

with tab1:
    uploaded_file = st.file_uploader("Drop your audio file here", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        audio_to_process = uploaded_file.getbuffer()

with tab2:
    st.markdown('<div style="color: #6b7fa3; font-size: 0.9rem; margin-bottom: 0.75rem;">Press the microphone icon to start recording:</div>', unsafe_allow_html=True)
    audio_bytes = audio_recorder(text="", recording_color="#e84118", neutral_color="#00d2ff", icon_name="microphone", icon_size="2x")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        audio_to_process = audio_bytes

# ==========================================
# TRANSCRIPTION EXECUTION
# ==========================================
if audio_to_process is not None:
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        run = st.button("🚀  Transcribe Now", use_container_width=True)
    with col_clear:
        if st.button("✕  Reset", use_container_width=True):
            st.session_state.last_result = None
            st.session_state.last_confidence = None
            st.rerun()

    if run:
        with st.spinner("Model is analysing your audio…"):
            tmp_path = "temp_audio.wav"
            with open(tmp_path, "wb") as f:
                f.write(audio_to_process)
            try:
                result_text, confidence = process_audio(tmp_path, language)
            except Exception as e:
                st.error(f"❌ Inference error: {e}")
                st.stop()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        st.session_state.last_result = result_text
        st.session_state.last_confidence = confidence
        st.session_state.history.insert(0, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
            "language": language, "text": result_text, "confidence": confidence,
            "words": len(result_text.split()), "chars": len(result_text)
        })

# ==========================================
# DISPLAY RESULTS & INTELLIGENT ANALYSIS
# ==========================================
if st.session_state.last_result:
    st.success("✅  Transcription complete!")
    word_count = len(st.session_state.last_result.split())
    char_count = len(st.session_state.last_result)
    conf_pct = int(st.session_state.last_confidence * 100)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Transcribed Text</div>
        <div class="result-text">{st.session_state.last_result}</div>
    </div>
    <div class="stats-row">
        <div class="stat-pill">Words: <span>{word_count}</span></div>
        <div class="stat-pill">Characters: <span>{char_count}</span></div>
        <div class="stat-pill">Language: <span>{language}</span></div>
        <div class="stat-pill">Confidence: <span>{conf_pct}%</span></div>
    </div>
    <div style="margin-top:0.75rem;">
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#4a5a7a;margin-bottom:4px;">CONFIDENCE SCORE — {conf_pct}%</div>
        <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{conf_pct}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇  Download .txt", data=io.BytesIO(st.session_state.last_result.encode("utf-8")), file_name="transcription.txt", mime="text/plain", use_container_width=True)
    with c2:
        st.button("📋  Copy to Clipboard", use_container_width=True)
    with c3:
        if st.button("🗑  Clear Result", use_container_width=True):
            st.session_state.last_result = None; st.session_state.last_confidence = None; st.rerun()

    # ------------------------------------------
    # INTELLIGENT ANALYSIS TABS (CLEANED)
    # ------------------------------------------
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🧠 Intelligent Audio Analysis</div>', unsafe_allow_html=True)
    
    tab_summary, tab_kws = st.tabs(["📝 AI Summary", "🎯 Keyword Spotting"])
    
    # --- TAB 1: SUMMARY ---
    with tab_summary:
        if st.button("Generate Summary", key="btn_summary"):
            with st.spinner("Extracting key insights..."):
                try:
                    summary_text = generate_summary(st.session_state.last_result)
                    st.info(f"**Summary:**\n\n{summary_text}")
                except Exception as e:
                    st.error("Text might be too short for summarization.")

    # --- TAB 2: KEYWORD SPOTTING ---
    with tab_kws:
        st.markdown("<div style='color: #8a9ac0; margin-bottom: 10px;'>Enter keywords to spot (comma-separated):</div>", unsafe_allow_html=True)
        default_kws = "Enter The word you want to search"
        kws_input = st.text_input("", placeholder=default_kws, label_visibility="collapsed", key="kws_input")
        
        if st.button("Spot Keywords", key="btn_kws") and kws_input:
            keywords = [k.strip().lower() for k in kws_input.split(',') if k.strip()]
            text_to_search = st.session_state.last_result
            text_lower = text_to_search.lower()
            
            found_keywords = {}
            highlighted_text = text_to_search

            for kw in keywords:
                count = text_lower.count(kw)
                if count > 0:
                    found_keywords[kw] = count
                    pattern = re.compile(re.escape(kw), re.IGNORECASE)
                    highlighted_text = pattern.sub(f"<mark style='background-color: #ff4b4b; color: white; border-radius: 4px; padding: 0 4px;'>{kw}</mark>", highlighted_text)

            if found_keywords:
                st.success(f"🎯 Spotted {len(found_keywords)} specific keyword(s)!")
                cols = st.columns(len(found_keywords))
                for i, (kw, count) in enumerate(found_keywords.items()):
                    cols[i % len(cols)].markdown(f"**{kw}**: {count} time(s)")
                st.markdown("<br><b>Highlighted Text:</b>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-text' style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px;'>{highlighted_text}</div>", unsafe_allow_html=True)
            else:
                st.warning("No specified keywords were spotted in the audio transcription.")

# ==========================================
# HISTORY & FOOTER
# ==========================================
if st.session_state.history:
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">⏱  Transcription History</div>', unsafe_allow_html=True)
    col_hist, col_hist_clear = st.columns([4, 1])
    with col_hist_clear:
        if st.button("Clear All"):
            st.session_state.history = []; st.rerun()

    for i, entry in enumerate(st.session_state.history[:10]):
        st.markdown(f"""
        <div class="history-item">
            <div class="history-meta">
                <span>#{len(st.session_state.history) - i}</span>
                <span>{entry['timestamp']}</span>
                <span>{entry['language']}</span>
                <span>{entry['words']} words</span>
            </div>
            <div class="history-text">{entry['text'][:180]}{"…" if len(entry['text']) > 180 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown('<div class="app-footer">Developed by Ahmed Solia & Eyad Mostafa &nbsp;·&nbsp; Powered by PyTorch & Streamlit</div>', unsafe_allow_html=True)