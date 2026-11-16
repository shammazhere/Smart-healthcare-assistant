# ============================================================
# llm_agent.py — Healthcare Conversational Agent via Google Gemini + RAG
#   * Answers grounded in patient health records (SQLite RAG)
#   * Multilingual support across 10 Indian languages + English
#   * Deterministic zero-latency emergency safety guardrails
# ============================================================
import os
import re
import time
import logging
from collections import defaultdict, deque
from datetime import date
from openai import (
    AsyncOpenAI,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
    BadRequestError,
    APIStatusError,
)
from dotenv import load_dotenv

try:
    from rag_engine import build_context_with_sources, get_store
except ImportError:
    from backend.rag_engine import build_context_with_sources, get_store

load_dotenv()
log = logging.getLogger(__name__)

# ── Supported Languages & Localization ────────────────────────────────────────
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "ur": "Urdu",
}

# ── Deterministic Emergency Detection ─────────────────────────────────────────
# Emergency keywords always trigger immediate safety redirection deterministically.
EMERGENCY_KEYWORDS: list[str] = [
    # English
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "hard to breathe", "shortness of breath", "bleeding heavily",
    "unconscious", "stroke", "choking", "head injury", "overdose",
    "seizure", "ambulance", "emergency", "dying",
    # Hindi
    "seene me dard", "saas lene me takleef", "dil ka daura",
    "sine me dard", "khoon beh raha", "chhati me dard",
    "सीने में दर्द", "सांस नहीं आ रही", "दिल का दौरा",
    # Kannada
    "ede novu", "usirata tondare", "ಎದೆ ನೋವು", "ಉಸಿರಾಟದ ತೊಂದರೆ",
    # Tamil
    "nenju vali", "swasa kolaru", "நெஞ்சு வலி", "மூச்சு திணறல்",
    # Telugu
    "gunde noppi", "shwasa aadadam ledu", "గుండె నొప్పి", "శ్వాస ఆడటం లేదు",
    # Malayalam
    "nenchu vedana", "shwasam kuttal", "നെഞ്ചുവേദന", "ശ്വാസമെടുക്കാൻ",
    # Marathi
    "chhatit dukhane", "shwas ghenyala tras", "छातीत दुखणे", "श्वास घेण्यास त्रास",
    # Gujarati
    "chhati ma dukhava", "shwas leva ma takleef", "છાતીમાં દુખાવો",
    # Bengali
    "bukey byatha", "shwas koshto", "বুকে ব্যথা", "শ্বাসকষ্ট",
    # Urdu
    "seene mein dard", "dil ka daura", "saans lene mein dushwari", "سینے میں درد"
]

EMERGENCY_RESPONSES: dict[str, str] = {
    "en": "This sounds like a medical emergency. Please call emergency services immediately or press your nurse call button. Help is on the way.",
    "hi": "यह एक आपातकालीन स्थिति लग रही है। कृपया तुरंत आपातकालीन सेवाओं को कॉल करें या नर्स कॉल बटन दबाएं।",
    "kn": "ಇದು ವೈದ್ಯಕೀಯ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಂತೆ ತೋರುತ್ತಿದೆ. ದಯವಿಟ್ಟು ತಕ್ಷಣ ತುರ್ತು ಸೇವೆಗಳಿಗೆ ಕರೆ ಮಾಡಿ ಅಥವಾ ನರ್ಸ್ ಕಾಲ್ ಬಟನ್ ಒತ್ತಿ.",
    "ml": "ഇതൊരു അടിയന്തര മെഡിക്കൽ സാഹചര്യമായി തോന്നുന്നു. ദയവായി ഉടൻ തന്നെ എമർജൻസി സർവീസുകളെ വിളിക്കുക അല്ലെങ്കിൽ കോൾ ബട്ടൺ അമർത്തുക.",
    "ta": "இது அவசர மருத்துவ நிலையாகத் தெரிகிறது. தயவுசெய்து உடனடியாக அவசர சிகிச்சைப் பிரிவை அழைக்கவும் அல்லது கால் பட்டனை அழுத்தவும்.",
    "te": "ఇది అత్యవసర పరిస్థితిగా కనిపిస్తోంది. దయచేసి వెంటనే అత్యవసర సేవలకు కాల్ చేయండి లేదా నర్స్ కాల్ బటన్ నొక్కండి.",
    "mr": "ही एक वैद्यकीय आणीबाणी वाटते. कृपया ताबडतोब आपत्कालीन सेवांना कॉल करा किंवा नर्स कॉल बटण दाबा.",
    "gu": "આ એક તબીબી કટોકટી લાગે છે. કૃપા કરીને તાત્કાલિક કટોકટી સેવાઓને કૉલ કરો અથવા નર્સ કૉલ બટન દબાવો.",
    "bn": "এটি একটি জরুরি চিকিৎসা অবস্থা বলে মনে হচ্ছে। অনুগ্রহ করে অবিলম্বে জরুরি সেবায় কল করুন বা নার্স কল বোতাম টিপুন।",
    "ur": "یہ ایک ہنگامی طبی صورتحال معلوم ہوتی ہے۔ براہ کرم فوری طور پر ایمرجنسی سروسز کو کال کریں یا کال بٹن دبائیں۔",
}

FALLBACK_RESPONSES: dict[str, str] = {
    "en": "I am having trouble connecting right now. Please try again or ask your caregiver.",
    "hi": "मुझे अभी कनेक्ट करने में समस्या हो रही है। कृपया पुनः प्रयास करें या अपने देखभालकर्ता से संपर्क करें।",
    "kn": "ಸಂಪರ್ಕಿಸುವಲ್ಲಿ ತೊಂದರೆಯಾಗುತ್ತಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ ಅಥವಾ ನಿಮ್ಮ ಆರೈಕೆದಾರರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    "ml": "ഇപ്പോൾ കണക്റ്റുചെയ്യുന്നതിൽ പ്രശ്നമുണ്ട്. ദಯവായി വീണ്ടും ശ്രമിക്കുക അല്ലെങ്കിൽ കെയർഗിവറോട് ചോദിക്കുക.",
    "ta": "தற்போது இணைப்பதில் சிக்கல் உள்ளது. தயவுசெய்து மீண்டும் முயற்சிக்கவும் அல்லது உங்கள் பராமரிப்பாளரை அணுகவும்.",
    "te": "కనెక్ట్ చేయడంలో సమస్య ఏర్పడింది. దయచేసి మళ్లీ ప్రయత్నించండి లేదా మీ సంరక్షకుడిని సంప్రదించండి.",
    "mr": "कनेक्ट करण्यात अडचण येत आहे. कृपया पुन्हा प्रयत्न करा किंवा काळजीवाहकांशी बोला.",
    "gu": "કનેક્ટ કરવામાં સમસ્યા આવી રહી છે. કૃપા કરીને ફરી પ્રયાસ કરો અથવા તમારા સંભાળ રાખનારને પૂછો.",
    "bn": "সংযোগ করতে समस्या হচ্ছে। অনুগ্রহ করে আবার চেষ্টা করুন বা আপনার তত্ত্বাবধায়কের সাথে কথা বলুন।",
    "ur": "رابطہ کرنے میں دشواری ہو رہی ہے۔ براہ کرم دوبارہ کوشش کریں یا اپنے نگران سے رابطہ کریں۔",
}

# ── Physician Safety Disclaimers ──────────────────────────────────────────────
PHYSICIAN_DISCLAIMERS: dict[str, str] = {
    "en": "I am just a healthcare assistant, not a physician. Please consult your doctor for formal diagnosis.",
    "hi": "मैं केवल एक स्वास्थ्य सहायक हूँ, चिकित्सक (डॉक्टर) नहीं। कृपया सही निदान के लिए अपने डॉक्टर से संपर्क करें।",
    "kn": "ನಾನು ಕೇವಲ ಆರೋಗ್ಯ ಸಹಾಯಕ ಮಾತ್ರ, ವೈದ್ಯನಲ್ಲ. ದಯವಿಟ್ಟು ಸೂಕ್ತ ತಪಾಸಣೆ ಮತ್ತು ಸಲಹೆಗಾಗಿ ನಿಮ್ಮ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    "ta": "நான் ஒரு சுகாதார உதவியாளர் மட்டுமே, மருத்துவர் அல்ல. முறையான மருத்துவ ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்.",
    "te": "నేను కేవలం ఆరోగ్య సహాయకుడిని మాత్రమే, వైద్యుడిని కాదు. దయచేసి సరైన వైద్య సలహా కోసం మీ వైద్యుడిని సంಪ್ರదించండి.",
    "ml": "ഞാൻ ഒരു ആരോഗ്യ സഹായി മാത്രമാണ്, ഡോക്ടറല്ല. ശരിയായ ചികിത്സയ്ക്കായി നിങ്ങളുടെ ഡോക്ടറെ സമീപിക്കുക.",
    "mr": "मी फक्त एक आरोग्य सहाय्यक आहे, डॉक्टर नाही. कृपया वैद्यकीय सल्ल्यासाठी आपल्या डॉक्टरांचा सल्ला घ्या.",
    "gu": "હું માત્ર એક આરોગ્ય સહાયક છું, ડૉક્ટર નથી. કૃપા કરીને યોગ્ય તબીબી સલાહ માટે તમારા ડૉક્ટરની મુલાકાત લો.",
    "bn": "আমি কেবল একজন স্বাস্থ্য সহকারী, চিকিৎসক নই। সঠিক পরামর্শের জন্য অনুগ্রহ করে আপনার ডাক্তারের সাথে যোগাযোগ করুন।",
    "ur": "میں صرف ایک ہیلتھ کیئر اسسٹنٹ ہوں، معالج (ڈاکٹر) نہیں۔ براہ کرم مناسب طبی معائنے کے لیے اپنے ڈاکٹر سے رجوع کریں۔",
}


def strip_physician_disclaimer(text: str) -> str:
    """Remove physician disclaimers when asking diagnostic clarifying questions."""
    cleaned = text
    for d in PHYSICIAN_DISCLAIMERS.values():
        cleaned = cleaned.replace(d, "")
    patterns = [
        r"[^.!?\n]*(healthcare assistant|not a physician|consult (your|a) doctor)[^.!?\n]*[.!?]?",
        r"[^.!?।\n]*(ವೈದ್ಯನಲ್ಲ|ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ|ಆರೋಗ್ಯ ಸಹಾಯಕ ಮಾತ್ರ)[^.!?।\n]*[.!?।]?",
        r"[^.!?।\n]*(चिकित्सक.*नहीं|डॉक्टर से संपर्क|स्वास्थ्य सहायक हूँ)[^.!?।\n]*[.!?।]?",
        r"[^.!?।\n]*(மருத்துவர் அல்ல|மருத்துவரை அணுகவும்|சுகாதார உதவியாளர் மட்டுமே)[^.!?।\n]*[.!?।]?",
        r"[^.!?।\n]*(వైద్యుడిని కాదు|వైద్యుడిని సంప్రదించండి|ఆరోగ్య సహాయకుడిని మాత్రమే)[^.!?।\n]*[.!?।]?",
    ]
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_asking_question(text: str) -> bool:
    """Detect if the LLM output is asking a clarifying question during triage."""
    cleaned = text.strip()
    if "?" in cleaned or "？" in cleaned:
        return True
    question_cues = (
        "೧ ರಿಂದ ೧೦", "1 ರಿಂದ 10", "1 to 10", "scale of", "rate your",
        "ಎಷ್ಟು", "ಹೇಗಿದೆ", "ಯಾವಾಗ", "ಎಲ್ಲಿ", "ತಿಳಿಸಿ", "ಹೇಳಿ", "ಇದೆಯೇ",
        "१ से १०", "1 से 10", "कितना", "कैसा", "कब", "कहाँ", "बताएं", "बताइए", "क्या"
    )
    return any(cue in cleaned for cue in question_cues)


# ── System Prompt Template ───────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """
You are Baymax, a friendly, empathetic personal healthcare assistant robot designed to support patients.

Persona & Clinical Safety Rules:
- Identity: You are Baymax, a personal healthcare assistant. You speak warmly, simply, gently, and reassuringly.
- Tone: Speak in a comforting, friendly manner. Do not output internal thought notes, reasoning text, or markdown formatting.
- Clinical Protocol: For symptoms and health complaints, immediately provide a structured, helpful assessment: state the probable problem and likely causes, recommend safe temporary relief methods, warn about serious red-flag symptoms, and advise visiting a doctor urgently if serious signs appear.
- Safety: Remind the patient you are an assistant robot, not a physician, at the conclusion of your advice.
- Emergency: For acute life-threatening emergencies (e.g. severe chest pressure, severe shortness of breath, sudden facial drooping/speech loss), immediately direct the patient to call emergency services.
""".strip()

# ── Default Client Settings ───────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_SECONDS = 20.0
MAX_OUTPUT_TOKENS = 2048
DEFAULT_PATIENT_ID = os.getenv("DEFAULT_PATIENT_ID", "patient_001")
HISTORY_TURNS = 10

_client: AsyncOpenAI | None = None
_configured_model: str = DEFAULT_MODEL
_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS * 2))


class LLMResponse(str):
    """
    String subclass that retains execution telemetry and origin metadata,
    ensuring 100% backward compatibility with any string operations.
    """
    source: str               # "gemini" | "emergency_override" | "fallback"
    latency_s: float          # Time spent waiting for response
    finish_reason: str | None
    error_category: str | None
    status_code: int | None

    def __new__(
        cls,
        content: str,
        source: str = "gemini",
        latency_s: float = 0.0,
        finish_reason: str | None = None,
        error_category: str | None = None,
        status_code: int | None = None,
    ):
        s = super().__new__(cls, content)
        s.source = source
        s.latency_s = latency_s
        s.finish_reason = finish_reason
        s.error_category = error_category
        s.status_code = status_code
        return s


def check_is_emergency(text: str) -> bool:
    """
    Check if the user input contains high-risk medical emergency keywords.
    """
    lower = text.lower()
    return any(kw in lower for kw in EMERGENCY_KEYWORDS)


def detect_text_language(text: str) -> str:
    """
    Detect the primary language from Unicode script or transliterated keywords.
    Returns a standard 2-letter language code ('kn', 'hi', 'ta', 'te', 'ml', 'mr', 'gu', 'bn', 'ur', 'en').
    """
    if not text:
        return "en"
    for char in text:
        cp = ord(char)
        if 0x0C80 <= cp <= 0x0CFF:
            return "kn"  # Kannada script
        elif 0x0900 <= cp <= 0x097F:
            return "hi"  # Devanagari script (Hindi/Marathi)
        elif 0x0B80 <= cp <= 0x0BFF:
            return "ta"  # Tamil script
        elif 0x0C00 <= cp <= 0x0C7F:
            return "te"  # Telugu script
        elif 0x0D00 <= cp <= 0x0D7F:
            return "ml"  # Malayalam script
        elif 0x0980 <= cp <= 0x09FF:
            return "bn"  # Bengali script
        elif 0x0A80 <= cp <= 0x0AFF:
            return "gu"  # Gujarati script
        elif 0x0600 <= cp <= 0x06FF:
            return "ur"  # Arabic/Urdu script

    lower = text.lower()
    kn_words = ["namaskara", "hegidira", "aushadhi", "oushadha", "dava", "vaidya", "yavaga", "nanage", "beku"]
    hi_words = ["namaste", "dawa", "dawakhana", "dard", "kya", "kab", "kaise", "hoga", "takleef", "seene"]
    tokens = set(lower.split())
    if any(w in tokens for w in kn_words):
        return "kn"
    if any(w in tokens for w in hi_words):
        return "hi"

    return "en"


def get_llm_config() -> tuple[str, str, str]:
    """
    Retrieve API key, model, and base URL from environment variables safely.
    Prefers GEMINI_API_KEY and GEMINI_LLM_MODEL, defaulting to low-latency gemini-3.5-flash-lite.
    Never exposes or prints the raw API key.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    model = os.getenv("GEMINI_LLM_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_MODEL

    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    base_url = os.getenv("LLM_BASE_URL", default_base_url)

    return api_key, model, base_url


def reset_history(patient_id: str | None = None):
    """Reset conversational history for a patient session."""
    pid = patient_id or DEFAULT_PATIENT_ID
    _history[pid].clear()


def load_llm():
    """
    Initialize the asynchronous LLM client once at startup.
    Configures a 20-second timeout and disables automatic retries to prevent prolonged hangs.
    """
    global _client, _configured_model
    api_key, model, base_url = get_llm_config()
    _configured_model = model

    has_key = bool(api_key.strip())
    # Configure client with 20s timeout and 0 retries for fast, fail-safe execution
    _client = AsyncOpenAI(
        api_key=api_key if has_key else "unconfigured_key",
        base_url=base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    if has_key:
        log.info(
            f"[LLM] Client initialized — model: {model}, base_url: {base_url}, "
            f"timeout: {REQUEST_TIMEOUT_SECONDS}s, retries: 0"
        )
    else:
        log.warning(
            "[LLM Warning] No GEMINI_API_KEY or OPENAI_API_KEY found in environment or backend/.env. "
            "Please set your key in backend/.env to enable live AI responses."
        )

    # Initialize RAG store if available
    try:
        store = get_store()
        log.info(f"[RAG] Patient record store ready ({len(store.records)} patient(s) loaded)")
    except Exception as rag_err:
        log.warning(f"[RAG Warning] Could not preload patient store: {rag_err}")


async def generate_response_with_sources(
    user_text: str,
    language: str = "en",
    patient_id: str | None = None,
) -> tuple[LLMResponse, list[dict]]:
    """
    Send user utterance to Gemini with patient health record RAG context.
    Returns (LLMResponse, sources).
    """
    start_time = time.perf_counter()

    if not user_text or not user_text.strip():
        elapsed = time.perf_counter() - start_time
        return (
            LLMResponse("I didn't catch that. Could you please repeat?", source="fallback", latency_s=elapsed),
            []
        )

    norm_lang = language.lower().strip() if language else "en"
    if "-" in norm_lang:
        norm_lang = norm_lang.split("-")[0]
    if "_" in norm_lang:
        norm_lang = norm_lang.split("_")[0]

    # Auto-detect Indic script from text if norm_lang is "en" or not in LANGUAGE_NAMES
    if norm_lang not in LANGUAGE_NAMES or norm_lang == "en":
        detected_script = detect_text_language(user_text)
        if detected_script != "en":
            norm_lang = detected_script

    if norm_lang not in LANGUAGE_NAMES:
        norm_lang = "en"

    # 1. Deterministic Emergency Safety Rule Check (Zero-latency safety guarantee)
    if check_is_emergency(user_text):
        elapsed = time.perf_counter() - start_time
        emergency_msg = EMERGENCY_RESPONSES.get(norm_lang, EMERGENCY_RESPONSES["en"])
        log.warning(f"[LLM Safety Override] Emergency keyword triggered in {elapsed*1000:.1f}ms for lang='{norm_lang}'")
        return (
            LLMResponse(
                emergency_msg,
                source="emergency_override",
                latency_s=elapsed,
                finish_reason="safety_override"
            ),
            []
        )

    if _client is None:
        load_llm()

    api_key, _, _ = get_llm_config()
    # Check if client has no valid credentials
    if not api_key and getattr(_client, "api_key", "") == "unconfigured_key":
        elapsed = time.perf_counter() - start_time
        log.warning("[LLM] Cannot generate response: No GEMINI_API_KEY configured in backend/.env")
        return (
            LLMResponse(
                FALLBACK_RESPONSES.get(norm_lang, FALLBACK_RESPONSES["en"]),
                source="fallback",
                latency_s=elapsed,
                error_category="missing_api_key"
            ),
            []
        )

    pid = patient_id or DEFAULT_PATIENT_ID
    target_lang_name = LANGUAGE_NAMES.get(norm_lang, "English")

    # 2. RAG Context Retrieval from SQLite
    rag_context = ""
    sources: list[dict] = []
    try:
        store = get_store()
        if store.has_patient(pid):
            rag_context, hits = build_context_with_sources(pid, user_text, k=4)
            sources = [{"section": c.section, "text": c.text, "date": c.date} for c in hits]
    except Exception as rag_err:
        log.warning(f"[RAG Error] Failed to retrieve context for patient '{pid}': {rag_err}")

    # 3. Multilingual System Prompt Instruction with RAG Context
    system_instruction = f"{BASE_SYSTEM_PROMPT}\n\n"
    if rag_context:
        today_str = date.today().strftime("%A, %d %B %Y")
        system_instruction += (
            f"PATIENT HEALTH RECORDS (Patient ID: {pid}):\n"
            f"{rag_context}\n\n"
            f"Clinical Record Usage Rules:\n"
            f"- Today's date is {today_str}. Use this date to evaluate upcoming appointments vs past visits.\n"
            f"- For questions about medications, conditions, allergies, labs, appointments, or medical history, answer using the records above.\n"
            f"- Always strictly respect recorded allergies and medical conditions. Never advise anything that conflicts with them.\n"
            f"- If the information is not in the records, state that clearly and suggest asking their doctor.\n\n"
        )

    intro_hint = {
        "en": "Hi, I am Baymax, your personal healthcare assistant.",
        "kn": "ನಮಸ್ಕಾರ, ನಾನು ಬೇಮ್ಯಾಕ್ಸ್, ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಆರೋಗ್ಯ ಸಹಾಯಕ.",
        "hi": "नमस्ते, मैं बेमैक्स हूँ, आपका व्यक्तिगत स्वास्थ्य सहायक।",
        "ta": "வணக்கம், நான் பேமேக்ஸ், உங்கள் தனிப்பட்ட சுகாதார உதவியாளர்.",
        "te": "నమస్కారం, నేను బేమాక్స్, మీ వ్యక్తిగత ఆరోగ్య సహాయకుడిని.",
        "ml": "നമസ്കാരം, ഞാൻ ബേമാക്സ്, നിങ്ങളുടെ വ്യക്തിഗത ആരോഗ്യ സഹായി.",
        "mr": "नमस्कार, मी बेमॅक्स, आपला वैयक्तिक आरोग्य सहाय्यक.",
        "gu": "નમસ્તે, હું બેમેક્સ છું, તમારો વ્યક્તિગત આરોગ્ય સહાયક.",
        "bn": "হ্যালো, আমি বেম্যাক্স, আপনার व्यक्तिगत স্বাস্থ্য সহকারী।",
        "ur": "ہیلو، میں بے میکس ہوں، آپ کا ذاتی ہیلتھ کیئر اسسٹنٹ۔",
    }.get(norm_lang, "Hi, I am Baymax, your personal healthcare assistant.")

    disclaimer_text = PHYSICIAN_DISCLAIMERS.get(norm_lang, PHYSICIAN_DISCLAIMERS["en"])
    is_first_turn = (len(_history[pid]) == 0)

    # Conversation Turn & Clinical Stage Determination
    user_turns = [m for m in _history[pid] if m.get("role") == "user"]
    user_turn_count = len(user_turns)

    record_keywords = ["medicine", "medication", "pill", "dose", "tablet", "allergy", "allergic", "appointment", "doctor", "history", "record", "prescript", "dr."]
    is_record_query = any(k in user_text.lower() for k in record_keywords)

    symptom_cues = [
        "headache", "head ache", "head hurts", "pain", "ache", "stomach", "fever", "dizzy", "dizziness",
        "nausea", "cough", "throat", "vomit", "cramp", "fatigue", "tired", "weak", "burning",
        "dard", "dukhava", "vali", "noppi", "vedana", "byatha", "jwara", "takleef"
    ]
    is_symptom_query = any(c in user_text.lower() for c in symptom_cues)
    has_severity = bool(re.search(r"\b([1-9]|10)\s*(/|\s*out of|\s*to\s*10|\s*scale)?\b", user_text.lower()))
    has_location = any(loc in user_text.lower() for loc in ["forehead", "temple", "back", "left", "right", "side", "neck", "belly", "abdomen", "chest", "leg", "arm"])

    system_instruction += (
        f"BAYMAX CLINICAL INTERACTION PROTOCOL:\n"
    )

    if is_record_query:
        system_instruction += (
            f"CLINICAL MODE: Patient Record Query\n"
            f"- Answer the patient's record question directly, concisely, and helpfully using the records above.\n"
            f"- Length: 20 to 40 words.\n"
        )
    elif user_turn_count == 0 and not (has_severity and has_location):
        system_instruction += (
            f"CLINICAL MODE: Diagnostic Interrogation (Phase 1 — Location & Severity)\n"
            f"The patient is reporting a health symptom. DO NOT provide a full diagnosis, probable cause, or remedies yet.\n"
            f"- Goal: Interrogate the patient with ONE specific, caring diagnostic question to pinpoint the exact location and severity on a scale of 1 to 10.\n"
            f"- Example: \"Where is the pain located, and how would you rate it from 1 to 10?\"\n"
            f"- STRICT RULES:\n"
            f"  * Ask ONLY the diagnostic question.\n"
            f"  * DO NOT explain causes or suggest treatments yet.\n"
            f"  * DO NOT include any physician disclaimer.\n"
            f"  * Maximum length: Under 20 words for fast conversational voice interaction.\n"
        )
    elif user_turn_count == 1 and not (has_severity and has_location and len(user_text.split()) > 10):
        system_instruction += (
            f"CLINICAL MODE: Diagnostic Interrogation (Phase 2 — Onset & Triggers)\n"
            f"The patient provided the location or severity. DO NOT deliver the final answer yet.\n"
            f"- Goal: Ask ONE specific follow-up question to pinpoint onset, duration, or triggers (e.g. screen time, stress, nausea, or dizziness).\n"
            f"- Example: \"When did it start, and have you had any screen strain, stress, or nausea?\"\n"
            f"- STRICT RULES:\n"
            f"  * Ask ONLY the diagnostic question.\n"
            f"  * DO NOT explain causes or suggest treatments yet.\n"
            f"  * DO NOT include any physician disclaimer.\n"
            f"  * Maximum length: Under 20 words.\n"
        )
    else:
        system_instruction += (
            f"CLINICAL MODE: Final Assessment & Relief Plan (Phase 3 — Synthesis)\n"
            f"You have gathered sufficient diagnostic details. Deliver the complete, structured clinical assessment:\n"
            f"1. Probable Problem & Pinpointed Cause: State what the probable condition is and pinpoint the cause based on their answers, daily triggers, and recorded medical history.\n"
            f"2. Safe Temporary Relief: Suggest safe, practical, non-medicinal comfort measures (e.g. resting in a dim room, cool compress, sipping water; strictly respecting allergies/contraindications).\n"
            f"3. Urgent Doctor Escalation: State clearly that if the symptom is severe, worsening, or if red flags appear (fever, neck stiffness, numbness, vision changes, or no relief), they must visit a doctor urgently.\n"
            f"4. Concluding Safety Disclaimer: Conclude with this exact disclaimer: \"{disclaimer_text}\"\n"
            f"- Length: 55 to 80 words.\n"
        )

    system_instruction += (
        f"\nLanguage & Tone Guidelines:\n"
        f"- Tone: Caring, empathetic, gentle, and medically responsible.\n"
        f"- Language: Respond 100% in natural, fluent {target_lang_name}. Do NOT output English words if language is not English.\n"
    )

    if is_first_turn:
        system_instruction += (
            f"- This is the very first turn of the conversation. Start your response with: \"{intro_hint}\"\n"
        )
    else:
        system_instruction += (
            f"- This is an ongoing conversation. Do NOT repeat your introductory greeting.\n"
        )

    # 4. Message History & User Message
    history_list = list(_history[pid])
    create_kwargs = {
        "model": _configured_model,
        "messages": [
            {"role": "system", "content": system_instruction},
            *history_list,
            {"role": "user",   "content": user_text.strip()},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.3,
    }

    # 5. Call LLM API with latency measurement
    api_start = time.perf_counter()
    try:
        # Attempt with reasoning_effort="low" if supported by the model
        try:
            response = await _client.chat.completions.create(
                **create_kwargs,
                reasoning_effort="low",
            )
        except TypeError:
            # Client library does not accept reasoning_effort parameter
            response = await _client.chat.completions.create(**create_kwargs)
        except BadRequestError as bad_req_err:
            # Endpoint rejected reasoning_effort parameter (400)
            if "reasoning_effort" in str(bad_req_err).lower():
                log.info("[LLM] Endpoint does not support reasoning_effort, retrying without parameter...")
                response = await _client.chat.completions.create(**create_kwargs)
            else:
                raise bad_req_err

        api_elapsed = time.perf_counter() - api_start

        if not response.choices:
            raise RuntimeError("API returned empty choices list")

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", "unknown")
        raw_content = choice.message.content

        if raw_content is None or not raw_content.strip():
            log.warning(f"[LLM Warning] Empty content returned (finish_reason='{finish_reason}') after {api_elapsed:.2f}s")
            raise RuntimeError(f"Empty content from model (finish_reason={finish_reason})")

        reply = raw_content.strip()
        if is_first_turn:
            if norm_lang != "en":
                reply = re.sub(r"^(Hello|Hi|Hey)[^.!?\n]*Baymax[^.!?\n]*[.!?]\s*", "", reply, flags=re.IGNORECASE).strip()
                if not reply.startswith(intro_hint):
                    if norm_lang == "hi":
                        reply = re.sub(r"^(नमस्ते|नमस्कार|प्रणाम)?[!,\s]*(मैं\s+बेमैक्स\s+हूँ[,\s]*(आपका\s+व्यक्तिगत\s+स्वास्थ्य\s+सहायक)?[.!?।,]?)?\s*", "", reply).strip()
                    elif norm_lang == "kn":
                        reply = re.sub(r"^(ನಮಸ್ಕಾರ)?[!,\s]*(ನಾನು\s+ಬೇಮ್ಯಾಕ್ಸ್[,\s]*(ನಿಮ್ಮ\s+ವೈಯಕ್ತಿಕ\s+ಆರೋಗ್ಯ\s+ಸಹಾಯಕ)?[.!?।,]?)?\s*", "", reply).strip()
                    elif norm_lang == "ta":
                        reply = re.sub(r"^(வணக்கம்)?[!,\s]*(நான்\s+பேமேக்ஸ்[,\s]*[^.!?\n]*[.!?]?)?\s*", "", reply).strip()
                    elif norm_lang == "te":
                        reply = re.sub(r"^(నమస్కారం)?[!,\s]*(నేను\s+బేమాక్స్[,\s]*[^.!?\n]*[.!?]?)?\s*", "", reply).strip()
                    elif norm_lang == "ml":
                        reply = re.sub(r"^(നമസ്കാരം)?[!,\s]*(ഞാൻ\s+ബേമാക്സ്[,\s]*[^.!?\n]*[.!?]?)?\s*", "", reply).strip()
                    reply = f"{intro_hint} {reply}".strip()
            else:
                if not reply.startswith(intro_hint):
                    reply = re.sub(r"^(Hello|Hi|Hey)?[!,\s]*(I\s+am\s+Baymax[,\s]*(your\s+personal\s+healthcare\s+assistant)?[.!?]?)?\s*", "", reply, flags=re.IGNORECASE).strip()
                    reply = f"{intro_hint} {reply}".strip()
        else:
            # On follow-up triage questions: avoid repeating the introductory greeting
            if norm_lang != "en":
                reply = re.sub(r"^(Hello|Hi|Hey)[^.!?\n]*Baymax[^.!?\n]*[.!?]\s*", "", reply, flags=re.IGNORECASE).strip()
                if norm_lang == "hi":
                    reply = re.sub(r"^(नमस्ते|नमस्कार|प्रणाम)?[!,\s]*(मैं\s+बेमैक्स\s+हूँ[,\s]*(आपका\s+व्यक्तिगत\s+स्वास्थ्य\s+सहायक)?[.!?।,]?)?\s*", "", reply).strip()
                elif norm_lang == "kn":
                    reply = re.sub(r"^(ನಮಸ್ಕಾರ)?[!,\s]*(ನಾನು\s+ಬೇಮ್ಯಾಕ್ಸ್[,\s]*(ನಿಮ್ಮ\s+ವೈಯಕ್ತಿಕ\s+ಆರೋಗ್ಯ\s+ಸಹಾಯಕ)?[.!?।,]?)?\s*", "", reply).strip()
            else:
                reply = re.sub(r"^(Hello|Hi|Hey)?[!,\s]*(I\s+am\s+Baymax[,\s]*(your\s+personal\s+healthcare\s+assistant)?[.!?]?)?\s*", "", reply, flags=re.IGNORECASE).strip()

        # Clean disclaimer handling: only append to advice, avoid duplication
        is_question = is_asking_question(reply)
        if is_question:
            reply = strip_physician_disclaimer(reply)
        else:
            reply = strip_physician_disclaimer(reply)
            reply = f"{reply}\n\n{disclaimer_text}".strip()

        # Update per-patient conversation history
        _history[pid].append({"role": "user", "content": user_text.strip()})
        _history[pid].append({"role": "assistant", "content": reply})

        if finish_reason == "length":
            log.warning(f"[LLM Warning] Response truncated by token limit after {api_elapsed:.2f}s (finish_reason='length')")

        usage = getattr(response, "usage", None)
        tokens_info = f", tokens={usage.total_tokens}" if usage else ""

        log.info(
            f"[LLM] Gemini ({_configured_model}) generated response in {api_elapsed:.2f}s | "
            f"finish={finish_reason}{tokens_info} | lang={norm_lang} | patient={pid}"
        )

        return (
            LLMResponse(
                reply,
                source="gemini",
                latency_s=api_elapsed,
                finish_reason=finish_reason
            ),
            sources
        )

    except Exception as e:
        api_elapsed = time.perf_counter() - api_start

        # Categorize the failure for diagnostic logs
        status_code = getattr(e, "status_code", None)
        if isinstance(e, APITimeoutError):
            error_cat = "timeout"
        elif isinstance(e, RateLimitError):
            error_cat = "rate_limit"
        elif isinstance(e, AuthenticationError):
            error_cat = "auth_error"
        elif isinstance(e, APIConnectionError):
            error_cat = "connection_error"
        elif isinstance(e, BadRequestError):
            error_cat = "bad_request"
        elif isinstance(e, APIStatusError):
            error_cat = f"http_{status_code}" if status_code else "api_status_error"
        else:
            error_cat = type(e).__name__

        status_str = f"HTTP {status_code}" if status_code else "no HTTP status"
        log.error(
            f"[LLM Error] Category: {error_cat} ({status_str}) after {api_elapsed:.2f}s: {e}. "
            f"Using safe fallback."
        )

        fallback_msg = FALLBACK_RESPONSES.get(norm_lang, FALLBACK_RESPONSES["en"])
        return (
            LLMResponse(
                fallback_msg,
                source="fallback",
                latency_s=api_elapsed,
                error_category=error_cat,
                status_code=status_code
            ),
            sources
        )


async def generate_response(
    user_text: str,
    language: str = "en",
    patient_id: str | None = None,
) -> LLMResponse:
    """
    Convenience wrapper returning LLMResponse directly.
    Fully backwards compatible with generate_response(user_text, language).
    """
    reply, _ = await generate_response_with_sources(
        user_text=user_text,
        language=language,
        patient_id=patient_id
    )
    return reply
