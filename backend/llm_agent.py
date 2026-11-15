# ============================================================
# llm_agent.py — Healthcare Conversational Agent via Google Gemini + RAG
#   * Answers grounded in patient health records (SQLite RAG)
#   * Multilingual support across 10 Indian languages + English
#   * Deterministic zero-latency emergency safety guardrails
# ============================================================
import os
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

# ── System Prompt Template ───────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """
You are HealthBot, a friendly, empathetic healthcare assistant robot designed to support patients.

Clinical Safety & Interaction Rules:
- Keep every response strictly under 40 words (short responses are essential for speech).
- Speak warmly, clearly, and reassuringly, as if talking to a recovering or elderly patient.
- Provide general wellness information only; NEVER diagnose, prescribe medication, or give specific treatment advice.
- Always remind users that you are an assistant robot and not a physician if they ask for diagnostic advice.
- For severe symptoms or medical emergencies: immediately direct the patient to call emergency services or press their call button.
- Topics supported: medication adherence reminders, hydration, sleep hygiene, gentle reassurance, and comfort check-ins.
""".strip()

# ── Default Client Settings ───────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_SECONDS = 20.0
MAX_OUTPUT_TOKENS = 600
DEFAULT_PATIENT_ID = os.getenv("DEFAULT_PATIENT_ID", "patient_001")
HISTORY_TURNS = 4

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

    system_instruction += (
        f"Language Instruction: Respond to the patient in natural, fluent {target_lang_name}. "
        f"If the patient uses mixed languages or code-switching, reply in their dominant language ({target_lang_name}). "
        f"Keep your complete response under 40 words."
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
        "temperature": 0.5,
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
