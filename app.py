import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import requests
import json
from datetime import datetime, timedelta
import tempfile
import glob
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import warnings
import logging
from typing import Optional, Dict, List, Tuple, Any
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
import pandas as pd
import plotly.express as px

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# ============================================================================
# STREAMLIT CLOUD OPTIMIZED CONFIG
# ============================================================================

class Config:
    """Optimized configuration for Streamlit Cloud"""
    DEBUG = False
    VERSION = "4.0-cloud"
    
    # Paths
    DOCUMENTS_PATH = "documents"
    
    # API
    GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1"
    GEMINI_MODELS = ['models/gemini-1.5-flash-latest', 'models/gemini-1.5-flash']
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 15  # Reduced for free tier
    REQUEST_TIMEOUT = 30
    
    # Retrieval (optimized for memory)
    CHUNK_SIZE = 800  # Reduced from 1000
    CHUNK_OVERLAP = 150  # Reduced from 200
    MIN_CHUNK_LENGTH = 50
    TOP_K_RESULTS = 3  # Reduced from 5
    
    # Memory
    MEMORY_WINDOW = 3  # Reduced from 5
    
    # Cache
    CACHE_TTL_SECONDS = 3600
    SIMILARITY_THRESHOLD = 0.95
    
    # Confidence
    HIGH_CONFIDENCE = 0.8
    MEDIUM_CONFIDENCE = 0.5
    
    # Languages
    SUPPORTED_LANGUAGES = ['vi', 'en']
    DEFAULT_LANGUAGE = 'vi'
    
    # Contact
    CONTACT_INFO = {
        'hotline': ['1900 5555 14', '0879 5555 14'],
        'email': 'tuyensinh@hcmulaw.edu.vn',
        'phone': '(028) 39400 989',
        'address': '2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM',
        'website': 'www.hcmulaw.edu.vn',
        'facebook': 'facebook.com/hcmulaw'
    }

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: str
    category: str = ""
    confidence: float = 0.0
    sources: List[str] = None
    feedback: Optional[int] = None

# ============================================================================
# UTILITIES
# ============================================================================

def init_session_state():
    """Initialize session state"""
    defaults = {
        "messages": [],
        "conversation_memory": ConversationBufferWindowMemory(
            k=Config.MEMORY_WINDOW,
            return_messages=True
        ),
        "first_visit": True,
        "request_count": 0,
        "last_request_time": datetime.now(),
        "error_count": 0,
        "pending_question": None,
        "language": Config.DEFAULT_LANGUAGE,
        "query_cache": {},
        "analytics": defaultdict(int),
        "feedback_data": [],
        "conversation_id": f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "show_analytics": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    text = " ".join(text.split())
    text = text[:max_length]
    dangerous = ['<script', 'javascript:', 'onerror=']
    for pattern in dangerous:
        text = text.replace(pattern, '')
    return text.strip()

def check_rate_limit() -> Tuple[bool, str]:
    """Rate limiting"""
    now = datetime.now()
    time_diff = (now - st.session_state.last_request_time).total_seconds()
    
    if time_diff < 60:
        if st.session_state.request_count >= Config.MAX_REQUESTS_PER_MINUTE:
            wait_time = int(60 - time_diff)
            return False, f"⏳ Vui lòng đợi {wait_time} giây"
    else:
        st.session_state.request_count = 0
        st.session_state.last_request_time = now
    
    st.session_state.request_count += 1
    return True, ""

def format_contact_info(language: str = 'vi') -> str:
    """Format contact info"""
    info = Config.CONTACT_INFO
    if language == 'vi':
        return f"""
📞 **Hotline:** {' hoặc '.join(info['hotline'])}
📧 **Email:** {info['email']}
☎️ **Điện thoại:** {info['phone']}
🌐 **Website:** {info['website']}
📍 **Địa chỉ:** {info['address']}
"""
    else:
        return f"""
📞 **Hotline:** {' or '.join(info['hotline'])}
📧 **Email:** {info['email']}
☎️ **Phone:** {info['phone']}
🌐 **Website:** {info['website']}
📍 **Address:** {info['address']}
"""

# ============================================================================
# EMBEDDINGS & VECTORSTORE (OPTIMIZED)
# ============================================================================

@st.cache_resource(show_spinner="🔄 Loading AI model... (first time: 2-3 min)")
def load_embeddings():
    """Load embeddings - optimized for Streamlit Cloud"""
    try:
        import sys, io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Suppress output
        
        embeddings = HuggingFaceEmbeddings(
            model_name="keepitreal/vietnamese-sbert",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="/tmp/sentence_transformers"
        )
        
        sys.stdout = old_stdout
        return embeddings
        
    except Exception as e:
        sys.stdout = old_stdout
        st.warning(f"⚠️ Could not load embeddings: {str(e)[:100]}")
        st.info("💡 Running in API-only mode (no RAG)")
        return None

def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """Download from Google Drive"""
    if not file_id:
        return False
    
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
            return True
        return False
    except:
        return False

def load_cached_vectorstore() -> Tuple[Optional[object], Dict]:
    """Load from Google Drive"""
    try:
        vectorstore_id = st.secrets.get("GDRIVE_VECTORSTORE_ID", None)
        metadata_id = st.secrets.get("GDRIVE_METADATA_ID", None)
    except:
        vectorstore_id = os.getenv("GDRIVE_VECTORSTORE_ID")
        metadata_id = os.getenv("GDRIVE_METADATA_ID")
    
    if not vectorstore_id or not metadata_id:
        return None, {}
    
    temp_dir = tempfile.mkdtemp()
    vectorstore_path = os.path.join(temp_dir, "vectorstore.pkl")
    metadata_path = os.path.join(temp_dir, "metadata.json")
    
    try:
        if not download_from_gdrive(vectorstore_id, vectorstore_path):
            return None, {}
        if not download_from_gdrive(metadata_id, metadata_path):
            return None, {}
        
        with open(vectorstore_path, 'rb') as f:
            vectorstore = pickle.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return vectorstore, metadata
        
    except Exception as e:
        return None, {}
    finally:
        try:
            if os.path.exists(vectorstore_path):
                os.remove(vectorstore_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

@st.cache_resource(show_spinner=False)
def initialize_vectorstore() -> Tuple[Optional[object], Dict]:
    """Initialize vectorstore - with proper status messages"""
    
    # Try Google Drive first
    with st.status("🔄 Loading vectorstore from Google Drive...") as status:
        try:
            vectorstore, metadata = load_cached_vectorstore()
            if vectorstore:
                status.update(label="✅ Vectorstore loaded from Google Drive", state="complete")
                return vectorstore, metadata.get('stats', {})
        except Exception as e:
            status.update(label=f"⚠️ Google Drive failed: {str(e)[:50]}", state="error")
    
    # Fallback: No vectorstore
    st.warning("⚠️ No vectorstore found - Running in API-only mode")
    st.info("💡 The chatbot will work but without document context")
    
    return None, {}

# ============================================================================
# GEMINI API
# ============================================================================

@st.cache_resource
def get_gemini_config() -> Optional[Dict]:
    """Get Gemini config"""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ Missing GEMINI_API_KEY!")
        st.info("Get your key at: https://aistudio.google.com/app/apikey")
        return None
    
    try:
        url = f"{Config.GEMINI_API_BASE}/models?key={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            available = [m['name'] for m in models_data.get('models', [])]
            
            selected = None
            for model in Config.GEMINI_MODELS:
                if model in available:
                    selected = model
                    break
            
            if selected:
                return {'api_key': api_key, 'model': selected}
        
        elif response.status_code == 400:
            st.error("❌ Invalid API key!")
        else:
            st.error(f"❌ API error: {response.status_code}")
        
        return None
        
    except Exception as e:
        st.error(f"❌ Cannot connect to Gemini: {e}")
        return None

def call_gemini_api(config: Dict, prompt: str) -> str:
    """Call Gemini API"""
    if not config:
        return "Error: No API config"
    
    url = f"{Config.GEMINI_API_BASE}/{config['model']}:generateContent?key={config['api_key']}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=Config.REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']
            return "Sorry, no valid response received."
        
        else:
            return "Sorry, the system is temporarily unavailable."
        
    except requests.exceptions.Timeout:
        return "Error: System is slow, please try again."
    except Exception as e:
        return f"Error: {str(e)}" if Config.DEBUG else "Sorry, an error occurred."

# ============================================================================
# QUESTION HANDLING
# ============================================================================

def classify_question(question: str, language: str = 'vi') -> str:
    """Classify question"""
    q_lower = question.lower()
    
    categories = {
        "vi": {
            "Tuyển sinh": ["tuyển sinh", "đăng ký", "hồ sơ", "điểm chuẩn", "xét tuyển"],
            "Học phí": ["học phí", "chi phí", "miễn giảm", "học bổng", "tiền"],
            "Chương trình đào tạo": ["chương trình", "môn học", "tín chỉ", "ngành", "khoa"],
            "Cơ sở vật chất": ["ký túc xá", "ktx", "thư viện", "phòng lab", "cơ sở"],
            "Việc làm": ["việc làm", "thực tập", "cơ hội", "nghề nghiệp"],
        },
        "en": {
            "Admission": ["admission", "enroll", "register", "application"],
            "Tuition": ["tuition", "fee", "cost", "scholarship"],
            "Program": ["program", "course", "curriculum", "major"],
            "Facilities": ["dormitory", "library", "lab", "facilities"],
            "Career": ["career", "job", "internship", "employment"],
        }
    }
    
    cats = categories.get(language, categories['vi'])
    
    for category, keywords in cats.items():
        if any(kw in q_lower for kw in keywords):
            return category
    
    return "Thông tin chung" if language == 'vi' else "General"

def get_conversation_context() -> str:
    """Get conversation context"""
    memory = st.session_state.conversation_memory
    try:
        history = memory.load_memory_variables({})
        messages = history.get('history', [])
        if not messages:
            return ""
        
        context_parts = []
        for msg in messages[-Config.MEMORY_WINDOW:]:
            role = "User" if msg.type == "human" else "Assistant"
            context_parts.append(f"{role}: {msg.content[:150]}")
        return "\n".join(context_parts)
    except:
        return ""

def create_prompt(question: str, context: str, category: str, language: str = 'vi') -> str:
    """Create prompt"""
    conv_context = get_conversation_context()
    conv_section = f"\nRECENT CONVERSATION:\n{conv_context}\n" if conv_context else ""
    
    if language == 'vi':
        return f"""Bạn là chuyên gia tư vấn {category.lower()} của Trường Đại học Luật TP. Hồ Chí Minh.
{conv_section}
THÔNG TIN THAM KHẢO:
{context}

THÔNG TIN LIÊN HỆ:
{format_contact_info('vi')}

CÂU HỎI: {question}

HƯỚNG DẪN:
1. Trả lời ngắn gọn, dễ hiểu (tối đa 150 từ)
2. Ưu tiên thông tin từ tài liệu tham khảo
3. Sử dụng thông tin liên hệ chính xác
4. Nếu không chắc, khuyến khích liên hệ trực tiếp
5. Sử dụng emoji để dễ đọc

Trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp:"""
    else:
        return f"""You are an admissions consultant for HCMC University of Law.
{conv_section}
REFERENCE INFO:
{context}

CONTACT:
{format_contact_info('en')}

QUESTION: {question}

GUIDELINES:
1. Keep response concise (max 150 words)
2. Prioritize reference information
3. Use accurate contact details
4. Encourage direct contact if uncertain
5. Use emojis for readability

Respond in English, friendly and professional:"""

def calculate_confidence(has_context: bool, answer_length: int) -> float:
    """Simple confidence calculation"""
    if not has_context:
        return Config.MEDIUM_CONFIDENCE
    
    # Based on answer length (sweet spot: 50-200 words)
    words = answer_length / 5  # rough estimate
    length_score = min(words / 100, 1.0)
    
    return min(0.7 + length_score * 0.3, 1.0)

def generate_answer(question: str, vectorstore: Optional[object], 
                   gemini_config: Dict) -> Tuple[str, str, float, List[str]]:
    """Generate answer"""
    
    language = st.session_state.language
    category = classify_question(question, language)
    sources = []
    
    try:
        if vectorstore:
            # RAG mode
            retriever = vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K_RESULTS})
            docs = retriever.invoke(question)
            context = "\n\n".join([f"[Source: {doc.metadata.get('source_file', 'Unknown')}]\n{doc.page_content}" 
                                   for doc in docs[:Config.TOP_K_RESULTS]])
            sources = [doc.metadata.get('source_file', 'Unknown') for doc in docs[:Config.TOP_K_RESULTS]]
            prompt = create_prompt(question, context, category, language)
            has_context = True
        else:
            # API-only mode
            prompt = f"""Bạn là chuyên gia tư vấn của Trường Đại học Luật TP. Hồ Chí Minh.

THÔNG TIN LIÊN HỆ:
{format_contact_info(language)}

CÂU HỎI: {question}

Trả lời chung và khuyến khích liên hệ để được tư vấn cụ thể.
Trả lời bằng {'tiếng Việt' if language == 'vi' else 'English'}, ngắn gọn:"""
            has_context = False
        
        # Generate
        answer = call_gemini_api(gemini_config, prompt)
        
        # Calculate confidence
        confidence = calculate_confidence(has_context, len(answer))
        
        # Update memory
        st.session_state.conversation_memory.save_context(
            {"input": question},
            {"output": answer}
        )
        
        # Update analytics
        st.session_state.analytics[category] += 1
        st.session_state.analytics['total_queries'] += 1
        
        return answer, category, confidence, sources
        
    except Exception as e:
        error_msg = f"""
❌ **Xin lỗi, đã có lỗi xảy ra**

Vui lòng liên hệ trực tiếp:

{format_contact_info(language)}
"""
        if Config.DEBUG:
            error_msg += f"\n\n_Debug: {str(e)[:100]}_"
        
        return error_msg, "Lỗi hệ thống", 0.0, []

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render header"""
    st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-container h1 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .header-container p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    </style>
    
    <div class="header-container">
        <h1>🤖 AI Chatbot Tư Vấn v4.0</h1>
        <p>Trường Đại học Luật TP. Hồ Chí Minh</p>
        <p>💬 Hỗ trợ 24/7 | 🎓 AI-Powered | ⚡ Smart Response</p>
    </div>
    """, unsafe_allow_html=True)

def get_confidence_badge(confidence: float) -> str:
    """Confidence badge"""
    if confidence >= Config.HIGH_CONFIDENCE:
        return '<span style="background:#4caf50;color:white;padding:4px 10px;border-radius:10px;font-size:0.75em;">✅ High ({:.0%})</span>'.format(confidence)
    elif confidence >= Config.MEDIUM_CONFIDENCE:
        return '<span style="background:#ff9800;color:white;padding:4px 10px;border-radius:10px;font-size:0.75em;">⚠️ Medium ({:.0%})</span>'.format(confidence)
    else:
        return '<span style="background:#f44336;color:white;padding:4px 10px;border-radius:10px;font-size:0.75em;">❌ Low ({:.0%})</span>'.format(confidence)

def get_category_badge(category: str) -> str:
    """Category badge"""
    colors = {
        "Tuyển sinh": "#1e88e5", "Admission": "#1e88e5",
        "Học phí": "#43a047", "Tuition": "#43a047",
        "Chương trình đào tạo": "#fb8c00", "Program": "#fb8c00",
        "Cơ sở vật chất": "#8e24aa", "Facilities": "#8e24aa",
        "Việc làm": "#e53935", "Career": "#e53935",
        "Thông tin chung": "#546e7a", "General": "#546e7a",
    }
    color = colors.get(category, "#546e7a")
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:12px;font-size:0.85em;font-weight:600;margin-bottom:8px;display:inline-block;">{category}</span>'

def render_sources(sources: List[str]):
    """Render sources"""
    if sources and sources != ["Cache"]:
        st.markdown("**📚 Nguồn tham khảo:**")
        for i, source in enumerate(sources, 1):
            st.caption(f"{i}. {source}")

def render_feedback_buttons(message_index: int):
    """Render feedback buttons"""
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        if st.button("👍", key=f"like_{message_index}"):
            record_feedback(message_index, 1)
    
    with col2:
        if st.button("👎", key=f"dislike_{message_index}"):
            record_feedback(message_index, -1)

def record_feedback(message_index: int, feedback: int):
    """Record feedback"""
    if message_index < len(st.session_state.messages):
        msg = st.session_state.messages[message_index]
        msg['feedback'] = feedback
        
        feedback_key = 'positive_feedback' if feedback > 0 else 'negative_feedback'
        st.session_state.analytics[feedback_key] += 1
        
        st.success("✅ Cảm ơn phản hồi!" if feedback > 0 else "📝 Chúng tôi sẽ cải thiện!")
        time.sleep(1)
        st.rerun()

def render_quick_questions():
    """Render quick questions"""
    language = st.session_state.language
    
    if language == 'vi':
        st.markdown("### 💡 Câu hỏi thường gặp")
        questions = [
            "📝 Thủ tục đăng ký xét tuyển như thế nào?",
            "💰 Học phí một năm là bao nhiêu?",
            "📚 Trường có những ngành học nào?",
            "🏠 Trường có ký túc xá không?",
            "🎓 Cơ hội việc làm sau khi tốt nghiệp?",
            "📞 Thông tin liên hệ của trường?"
        ]
    else:
        st.markdown("### 💡 Frequently Asked Questions")
        questions = [
            "📝 How to apply for admission?",
            "💰 What is the annual tuition?",
            "📚 What programs are offered?",
            "🏠 Is there a dormitory?",
            "🎓 Career opportunities?",
            "📞 Contact information?"
        ]
    
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            clean_q = ' '.join(q.split()[1:])
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_question = clean_q
                st.session_state.first_visit = False
                st.rerun()

def render_analytics():
    """Render analytics"""
    st.markdown("### 📊 Thống kê")
    
    analytics = st.session_state.analytics
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tổng câu hỏi", analytics.get('total_queries', 0))
    
    with col2:
        positive = analytics.get('positive_feedback', 0)
        negative = analytics.get('negative_feedback', 0)
        total_fb = positive + negative
        satisfaction = (positive / total_fb * 100) if total_fb > 0 else 0
        st.metric("Độ hài lòng", f"{satisfaction:.0f}%")
    
    with col3:
        st.metric("Phản hồi (+/-)", f"{positive}/{negative}")

def render_sidebar(stats: Dict):
    """Render sidebar"""
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")
        
        # Language
        lang_options = {'vi': '🇻🇳 Tiếng Việt', 'en': '🇬🇧 English'}
        selected_lang = st.selectbox(
            "Ngôn ngữ",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=0 if st.session_state.language == 'vi' else 1
        )
        
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
        
        # Status
        with st.expander("📊 Trạng thái", expanded=False):
            st.success("✅ Gemini API: Active")
            
            if stats:
                st.info(f"📁 Files: {stats.get('processed_files', 0)}")
                st.info(f"📦 Chunks: {stats.get('total_chunks', 0)}")
            else:
                st.warning("⚠️ No vectorstore (API-only mode)")
            
            memory_turns = len(st.session_state.conversation_memory.chat_memory.messages)
            st.info(f"🧠 Memory: {memory_turns} turns")
        
        # Actions
        st.markdown("### 🔧 Thao tác")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.clear()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_memory.clear()
                st.session_state.first_visit = True
                st.rerun()
        
        # Export
        st.markdown("### 💾 Export")
        
        if st.session_state.messages:
            export_data = export_chat_txt()
            st.download_button(
                label="📥 Download (.txt)",
                data=export_data,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            total_msgs = len(st.session_state.messages)
            user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
            st.caption(f"📊 {total_msgs} messages ({user_msgs} questions)")
        else:
            st.info("No chat history")
        
        # Analytics toggle
        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.show_analytics = not st.session_state.get('show_analytics', False)
            st.rerun()
        
        # Contact
        st.markdown("---")
        st.markdown("### 📞 Contact")
        st.markdown(format_contact_info(st.session_state.language))
        
        # Footer
        st.markdown("---")
        st.caption(f"🤖 Chatbot v{Config.VERSION}")
        st.caption("Features: Memory | Feedback | Analytics")

def export_chat_txt() -> str:
    """Export as text"""
    content = "=" * 60 + "\n"
    content += "CHAT HISTORY - AI CHATBOT v4.0\n"
    content += "HCMC University of Law\n"
    content += f"Exported: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
    content += "=" * 60 + "\n\n"
    
    for i, msg in enumerate(st.session_state.messages, 1):
        role = "USER" if msg["role"] == "user" else "CHATBOT"
        content += f"{role}:\n{msg['content']}\n"
        
        if msg["role"] == "assistant":
            if "category" in msg:
                content += f"Category: {msg['category']}\n"
            if "confidence" in msg:
                content += f"Confidence: {msg['confidence']:.0%}\n"
            if "sources" in msg and msg['sources']:
                content += f"Sources: {', '.join(msg['sources'])}\n"
        
        content += "\n" + "-" * 60 + "\n\n"
    
    content += "\n" + format_contact_info()
    return content

def render_footer():
    """Render footer"""
    st.markdown("---")
    info = Config.CONTACT_INFO
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: #f5f7fa; border-radius: 12px;">
        <h4>🏛️ Trường Đại học Luật TP. Hồ Chí Minh</h4>
        <p>📍 {info['address']}</p>
        <p>📞 {' | '.join(info['hotline'])} | ☎️ {info['phone']}</p>
        <p>📧 {info['email']} | 🌐 {info['website']}</p>
        <p style="margin-top: 1rem; opacity: 0.7; font-size: 0.85em;">
            🚀 Powered by Gemini AI | Version {Config.VERSION}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application - Streamlit Cloud optimized"""
    
    st.set_page_config(
        page_title="AI Chatbot v4.0 - HCMC Law University",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_dotenv()
    init_session_state()
    
    render_header()
    
    # Initialize with proper error handling
    st.info("🔄 Initializing system...")
    
    # Step 1: Gemini API (critical)
    gemini_config = get_gemini_config()
    if not gemini_config:
        st.error("❌ Cannot connect to Gemini API. Please check your API key in Secrets.")
        st.stop()
    
    st.success("✅ Gemini API connected")
    
    # Step 2: Vectorstore (optional)
    try:
        vectorstore, stats = initialize_vectorstore()
        if vectorstore:
            st.success(f"✅ Vectorstore loaded ({stats.get('total_chunks', 0)} chunks)")
        else:
            st.info("💡 Running in API-only mode (no document context)")
    except Exception as e:
        st.warning(f"⚠️ Vectorstore failed: {str(e)[:100]}")
        st.info("💡 Continuing in API-only mode")
        vectorstore = None
        stats = {}
    
    # Clear initialization messages
    time.sleep(1)
    
    # Render sidebar
    render_sidebar(stats)
    
    # Show analytics if toggled
    if st.session_state.get('show_analytics', False):
        render_analytics()
        st.markdown("---")
    
    # First visit
    if not st.session_state.messages and st.session_state.first_visit:
        render_quick_questions()
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem; border-radius: 12px; color: white; margin-top: 1.5rem;">
            <h4>💡 New Features v4.0:</h4>
            <ul style="margin: 0.5rem 0;">
                <li>🧠 <strong>Conversation Memory:</strong> Remembers context</li>
                <li>📊 <strong>Confidence Scores:</strong> Shows answer reliability</li>
                <li>💬 <strong>Feedback System:</strong> Rate responses with 👍/👎</li>
                <li>📚 <strong>Source Attribution:</strong> See referenced documents</li>
                <li>🌐 <strong>Multi-language:</strong> Vietnamese & English support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Badges
                badge_html = get_category_badge(msg.get("category", ""))
                if msg.get("confidence", 0) > 0:
                    badge_html += " " + get_confidence_badge(msg["confidence"])
                st.markdown(badge_html, unsafe_allow_html=True)
            
            # Content
            st.markdown(msg["content"])
            
            # Sources
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])
            
            # Feedback
            if msg["role"] == "assistant" and msg.get("feedback") is None:
                render_feedback_buttons(idx)
            elif msg["role"] == "assistant" and msg.get("feedback"):
                fb_text = "👍 Helpful" if msg["feedback"] > 0 else "👎 Needs improvement"
                st.caption(f"💬 Your feedback: {fb_text}")
    
    # Handle input
    user_input = None
    
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    
    if not user_input:
        placeholder = "💬 Type your question..." if st.session_state.language == 'en' else "💬 Nhập câu hỏi..."
        user_input = st.chat_input(placeholder)
    
    # Process input
    if user_input:
        user_input = sanitize_input(user_input)
        
        if not user_input:
            st.warning("⚠️ Please enter a valid question")
            st.rerun()
        
        # Rate limit
        can_proceed, rate_msg = check_rate_limit()
        if not can_proceed:
            st.error(rate_msg)
            st.rerun()
        
        # Mark as not first visit
        st.session_state.first_visit = False
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Rerun to display
        st.rerun()
    
    # Generate answer if needed
    if (st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "user" and
        (len(st.session_state.messages) == 1 or 
         st.session_state.messages[-2]["role"] == "assistant")):
        
        last_question = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    answer, category, confidence, sources = generate_answer(
                        last_question, vectorstore, gemini_config
                    )
                    
                    # Display badges
                    badge_html = get_category_badge(category)
                    if confidence > 0:
                        badge_html += " " + get_confidence_badge(confidence)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        render_sources(sources)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "category": category,
                        "confidence": confidence,
                        "sources": sources,
                        "timestamp": datetime.now().isoformat(),
                        "feedback": None
                    })
                    
                    # Reset error count
                    st.session_state.error_count = 0
                    
                    # Feedback buttons
                    render_feedback_buttons(len(st.session_state.messages) - 1)
                    
                except Exception as e:
                    st.session_state.error_count += 1
                    
                    lang = st.session_state.language
                    error_message = f"""
❌ **Sorry, an error occurred**

Please try again or contact us:

{format_contact_info(lang)}
"""
                    if Config.DEBUG:
                        error_message += f"\n\n_Debug: {str(e)[:200]}_"
                    
                    st.error(error_message)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "category": "System Error",
                        "confidence": 0.0,
                        "sources": [],
                        "timestamp": datetime.now().isoformat(),
                        "feedback": None
                    })
                    
                    if st.session_state.error_count >= 3:
                        st.warning("⚠️ Multiple errors detected. Try refreshing the page.")
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()
