import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import gradio as gr
import requests
import time
import os
import struct
import base64
import re
from dotenv import load_dotenv

# --- Config ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# TTS Configuration
TTS_VOICES = {
    "English": {"voice_name": "Kore", "language_code": "en-US"},
    "Hindi (हिंदी)": {"voice_name": "Fenrir", "language_code": "hi-IN"},
    "Marathi (मराठी)": {"voice_name": "Leda", "language_code": "mr-IN"},
}
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{TTS_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --- Multilingual UI Strings ---
UI_STRINGS = {
    "English": {
        "title": "## 💰 NOVARYN",
        "desc": "Upload your bank statement CSV (optional) and enter fixed expenses/income. Then chat with your financial assistant!",
        "lang_label": "Select Language / भाषा चुनें",
        "csv_label": "Upload Bank Statement CSV (Optional - e.g., columns: Date, Description, Amount)",
        "income_label": "Average Monthly Income (₹)",
        "rent_label": "Rent (₹)",
        "emi_label": "EMI/Loan (₹)",
        "debt_label": "Debt Payments (₹)",
        "food_label": "Fixed Food/Groceries (₹)",
        "subscription_label": "Subscription (₹)",
        "custom_label": "Other Fixed Expenses (e.g., Gym 1500 or PetCare:500)",
        "upload_btn": "Upload & Get Initial Financial Overview",
        "output_label": "Initial Financial Overview",
        "chat_title": "## 🤖 Financial Chatbot",
        "question_label": "Ask your financial advisor",
        "question_placeholder": "Type your question here, e.g., 'What are good strategies for saving 10% more?'",
        "chat_output_label": "Advisor Response",
        "speak_btn": "🔊 Speak Response",
        "submit_btn": "Submit Question",
        "tts_status": "TTS Status:",
    },
    "Hindi (हिंदी)": {
        "title": "## 💰 NOVARYN",
        "desc": "अपना बैंक स्टेटमेंट CSV अपलोड करें (वैकल्पिक) और निश्चित खर्च/आय दर्ज करें। फिर अपने वित्तीय सहायक से चैट करें!",
        "lang_label": "भाषा चुनें / Select Language",
        "csv_label": "बैंक स्टेटमेंट CSV अपलोड करें (वैकल्पिक - उदा. कॉलम: दिनांक, विवरण, राशि)",
        "income_label": "औसत मासिक आय (₹)",
        "rent_label": "किराया (₹)",
        "emi_label": "ईएमआई/ऋण (₹)",
        "debt_label": "ऋण भुगतान (₹)",
        "food_label": "निश्चित भोजन/किराना (₹)",
        "subscription_label": "सदस्यता (₹)",
        "custom_label": "अन्य निश्चित खर्च (उदा. जिम 1500 या पेटकेअर:500)",
        "upload_btn": "अपलोड करें और प्रारंभिक वित्तीय अवलोकन प्राप्त करें",
        "output_label": "प्रारंभिक वित्तीय अवलोकन",
        "chat_title": "## 🤖 वित्तीय चैटबॉट",
        "question_label": "अपने वित्तीय सलाहकार से पूछें",
        "question_placeholder": "अपना प्रश्न यहाँ टाइप करें, उदा. '10% अधिक बचत करने की अच्छी रणनीतियाँ क्या हैं?'",
        "chat_output_label": "सलाहकार प्रतिक्रिया",
        "speak_btn": "🔊 प्रतिक्रिया बोलें",
        "submit_btn": "प्रश्न सबमिट करें",
        "tts_status": "टीटीएस स्थिति:",
    },
    "Marathi (मराठी)": {
        "title": "## 💰 NOVARYN",
        "desc": "तुमचे बँक स्टेटमेंट CSV अपलोड करा (ऐच्छिक) आणि निश्चित खर्च/उत्पन्न प्रविष्ट करा. त्यानंतर तुमच्या आर्थिक सहाय्यकाशी गप्पा मारा!",
        "lang_label": "भाषा निवडा / Select Language",
        "csv_label": "बँक स्टेटमेंट CSV अपलोड करा (ऐच्छिक - उदा. स्तंभ: तारीख, वर्णन, रक्कम)",
        "income_label": "सरासरी मासिक उत्पन्न (₹)",
        "rent_label": "भाडे (₹)",
        "emi_label": "ईएमआय/कर्ज (₹)",
        "debt_label": "कर्ज भरणा (₹)",
        "food_label": "निश्चित भोजन/किराणा (₹)",
        "subscription_label": "सदस्यता (₹)",
        "custom_label": "इतर निश्चित खर्च (उदा. जिम 1500 किंवा पेटकेअर:500)",
        "upload_btn": "अपलोड करा आणि प्रारंभिक आर्थिक विहंगावलोकन मिळवा",
        "output_label": "प्रारंभिक आर्थिक विहंगावलोकन",
        "chat_title": "## 🤖 आर्थिक चॅटबॉट",
        "question_label": "तुमच्या आर्थिक सल्लागाराला विचारा",
        "question_placeholder": "तुमचा प्रश्न येथे टाइप करा, उदा. '10% अधिक बचत करण्याची चांगली रणनीती कोणती आहेत?'",
        "chat_output_label": "सलाहगार प्रतिसाद",
        "speak_btn": "🔊 प्रतिसाद बोला",
        "submit_btn": "प्रश्न सबमिट करा",
        "tts_status": "टीटीएस स्थिती:",
    }
}

# --- ML: Expense Prediction ---
def generate_synthetic_data():
    """Generates 24 months of mock expenses for training."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=24, freq='MS')
    expenses = np.random.normal(loc=15000, scale=3000, size=24).round(2)
    data = pd.DataFrame({
        'month_start_date': dates,
        'monthly_expense': expenses
    })
    data['prev_month_expense'] = data['monthly_expense'].shift(1)
    return data.dropna().reset_index(drop=True)

def train_and_save_model():
    """Trains Linear Regression on synthetic data and saves the model."""
    data = generate_synthetic_data()
    X = data[['prev_month_expense']]
    Y = data['monthly_expense']
    model = LinearRegression()
    model.fit(X, Y)
    joblib.dump(model, 'monthly_expense_predictor.joblib')
    return model

# Train model when script runs
model = train_and_save_model()

def predict_variable_expense(csv_file):
    """Predicts next month's variable expense based on the uploaded CSV total."""
    try:
        if csv_file is None:
            return 0.0

        df = pd.read_csv(csv_file.name)
        # Sum of 'Amount' is treated as last month's variable expense
        prev_month_expense_total = df['Amount'].sum()

        model = joblib.load('monthly_expense_predictor.joblib')

        predicted = model.predict([[prev_month_expense_total]])[0]
        return predicted
    except Exception as e:
        if csv_file is not None:
            return f"Error processing CSV for prediction: {e}"
        return 0.0

# --- Multilingual Number Utility ---
def translate_numbers_to_devanagari(text):
    """Translates Hindu-Arabic numerals (0-9) in a string to Devanagari numerals."""
    mapping = str.maketrans('0123456789', '०१२३४५६७८९')
    return text.translate(mapping)

# --- TTS Utility Functions ---

def _create_wav_header(sample_rate, channels, bits_per_sample, data_size):
    """Generates a proper WAV file header from audio parameters."""
    header = b'RIFF'
    header += struct.pack('<I', 36 + data_size)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)
    header += struct.pack('<H', 1) # AudioFormat (1=PCM)
    header += struct.pack('<H', channels)
    header += struct.pack('<I', sample_rate)
    byte_rate = sample_rate * channels * bits_per_sample // 8
    header += struct.pack('<I', byte_rate)
    block_align = channels * bits_per_sample // 8
    header += struct.pack('<H', block_align)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)
    return header

def create_wav_bytes_from_pcm(pcm_data: bytes, sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Converts raw PCM audio bytes to complete WAV file bytes."""
    wav_header = _create_wav_header(sample_rate, channels, bits_per_sample, len(pcm_data))
    return wav_header + pcm_data

def speak_response(full_markdown_text, language):
    """Generates speech for the advisor's response using Gemini TTS API."""

    yield gr.update(value=None, visible=False, interactive=False), "TTS Status: Extracting text..."

    try:
        # Extract core text, stripping surrounding report/chat context and markdown
        core_text_to_speak = full_markdown_text
        if "💬 Advisor:" in full_markdown_text:
            start_tag = '💬 Advisor:\n'
            start_index = full_markdown_text.find(start_tag) + len(start_tag)
            end_tag = '\n\n**Cited Sources:**'
            end_index = full_markdown_text.find(end_tag, start_index)
            core_text_to_speak = full_markdown_text[start_index:end_index].strip() if end_index != -1 else full_markdown_text[start_index:].strip()

        core_text_to_speak = re.sub(r'[#*`]', '', core_text_to_speak).strip()

        if not core_text_to_speak:
            yield gr.update(value=None, visible=False, interactive=False), "TTS Status: Error - No meaningful text available to speak. Generate a response first."
            return

        # Text truncation check
        MAX_CHARS = 800
        if len(core_text_to_speak) > MAX_CHARS:
            original_len = len(core_text_to_speak)
            core_text_to_speak = core_text_to_speak[:MAX_CHARS] + "..."
            yield gr.update(value=None, visible=False, interactive=False), f"TTS Status: Warning! Text truncated from {original_len} to {MAX_CHARS} characters."

    except Exception as e:
        yield gr.update(value=None, visible=False, interactive=False), f"TTS Status: Error parsing text: {e}"
        return

    # API preparation
    voice_config = TTS_VOICES.get(language, TTS_VOICES["English"])
    payload = {
        "contents": [{ "parts": [{ "text": core_text_to_speak }] }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": { "voiceName": voice_config["voice_name"] }
                }
            }
        },
        "model": TTS_MODEL
    }

    # API Call with Exponential Backoff
    max_retries = 3
    base_delay = 1.0
    API_TIMEOUT = 30

    yield gr.update(value=None, visible=False, interactive=False), "TTS Status: Calling Gemini TTS API..."

    for attempt in range(max_retries):
        try:
            response = requests.post(
                TTS_API_URL,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()

            part = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
            audio_data_base64 = part.get('inlineData', {}).get('data')
            mime_type = part.get('inlineData', {}).get('mimeType')

            if not audio_data_base64 or not mime_type or not mime_type.startswith("audio/L16"):
                yield gr.update(value=None, visible=False, interactive=False), f"TTS Status: Error - TTS failed to return valid audio data ({mime_type})."
                return

            sample_rate = 24000
            yield gr.update(value=None, visible=False, interactive=False), "TTS Status: Processing audio data..."

            pcm_data = base64.b64decode(audio_data_base64)
            wav_bytes = create_wav_bytes_from_pcm(pcm_data, sample_rate)
            wav_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            data_url = f"data:audio/wav;base64,{wav_base64}"

            yield gr.update(value=data_url, visible=True, interactive=True), "TTS Status: Speech generated successfully! Ready to play."
            return

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                yield gr.update(value=None, visible=False, interactive=False), f"TTS Status: Request timed out. Retrying in {delay:.1f}s..."
                time.sleep(delay)
            else:
                yield gr.update(value=None, visible=False, interactive=False), f"TTS Status: Error: Request timed out after {max_retries} attempts."
                return
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                yield gr.update(value=None, visible=False, interactive=False), f"TTS Status: API Error {response.status_code}. Retrying in {delay:.1f}s..."
                time.sleep(delay)
            else:
                yield gr.update(value=None, visible=False, interactive=False), f"TTS Status: API HTTP Error ({response.status_code}): {e}"
                return
        except Exception as e:
            yield gr.update(value=None, visible=False, interactive=False), f"TTS Status: General API Error: {type(e).__name__}: {e}"
            return

    yield gr.update(value=None, visible=False, interactive=False), "TTS Status: Error - Max retries reached or unhandled failure."


# --- Input Parsing Utility ---
def parse_custom_expenses(custom_expenses_str):
    """
    Parses a string of custom expenses (e.g., 'Gym 1500, PetCare:500').
    Accepts space, comma, or newline separation.
    """
    expenses = {}
    if not custom_expenses_str:
        return expenses

    cleaned_str = custom_expenses_str.replace('\n', ' ').replace(',', ' ').replace(':', ' ').strip()
    # Pattern: Name (text) followed by Value (number)
    pattern = r'([^\d\s]+(?:\s[^\d\s]+)*)\s+(\d+(?:\.\d+)?)'

    matches = re.findall(pattern, cleaned_str, re.IGNORECASE)

    for name_raw, value_str in matches:
        name = name_raw.strip()
        try:
            value = float(value_str)
            if name and value > 0:
                expenses[name] = value
        except ValueError:
            continue

    return expenses


# --- Google AI Chatbot ---
def ask_advisor(question, fixed_expenses, income, csv_file, language="English"):
    """Core function to get financial advice from Gemini."""
    try:
        predicted_variable = predict_variable_expense(csv_file)

        if isinstance(predicted_variable, str):
            return predicted_variable

        prediction_is_missing = (predicted_variable == 0.0 and csv_file is None)

        # Financial Calculations
        total_fixed = sum(fixed_expenses.values())
        total_expected = total_fixed + predicted_variable
        surplus = income - total_expected

        # Language formatting setup
        if language in ["Hindi (हिंदी)", "Marathi (मराठी)"]:
            currency_symbol = "₹"
            number_translator = translate_numbers_to_devanagari
        else:
            currency_symbol = "₹"
            number_translator = lambda x: x

        def format_currency(amount):
            formatted = f"{currency_symbol}{amount:,.2f}"
            return number_translator(formatted)

        # Status Summary and Allocation
        advice_summary = ""
        predicted_expense_text = f"💰 Predicted Variable Expense: {format_currency(predicted_variable)}"

        if prediction_is_missing:
            advice_summary = "ℹ️ Note: Variable expenses (predicted 0) are unbudgeted as no bank statement was uploaded. Surplus calculation relies only on fixed expenses."
        elif total_expected > income:
            advice_summary = "⚠️ Warning: Predicted total expense exceeds income! Immediate action required."
        elif total_expected > income * 0.8:
            advice_summary = "🔔 Caution: Total expense close to income (80% usage). Consider tightening variable spending."
        else:
            advice_summary = "✅ Good news! Total predicted expense is within safe limits. Focus on savings goals."

        # Allocation Advice (30/30/30/10 Rule)
        allocation_advice = ""
        emergency_allocation = 0.0
        retirement_allocation = 0.0

        if surplus > 0:
            emergency_allocation = surplus * 0.30
            retirement_allocation = surplus * 0.30
            goal_allocation = surplus * 0.30
            flex_allocation = surplus * 0.10

            allocation_advice = f"""
### Monthly Surplus Allocation (30/30/30/10 Rule)
| Category | Percentage | Amount | Priority |
| :--- | :--- | :--- | :--- |
| **Emergency Fund** | 30% | {format_currency(emergency_allocation)} | High (Liquid Savings/FDs) |
| **Retirement/L.T.** | 30% | {format_currency(retirement_allocation)} | Medium (SIPs/Mutual Funds/PPF) |
| **Specific Goals** | 30% | {format_currency(goal_allocation)} | Medium (Medium-risk instruments) |
| **Flexible Savings** | 10% | {format_currency(flex_allocation)} | Low (Short-term use/Premiums) |
"""
        else:
            allocation_advice = "### Monthly Surplus Allocation\nThere is no positive surplus to allocate. Focus on reducing current fixed or variable expenses."

        # Fixed Expense Breakdown
        fixed_breakdown = "\n".join([f"- {name.capitalize()}: ₹{amount:,.2f}" for name, amount in fixed_expenses.items()])
        fixed_breakdown = translate_numbers_to_devanagari(fixed_breakdown)

        # System Instruction for the AI
        system_prompt = f"""You are a helpful, friendly, and professional financial advisor. Your advice must be practical, affordable, and based on the user's income (₹{{income:.2f}}). Respond entirely in {language}. Keep all responses short, concise, and structured using bulleted lists. Do not use long paragraphs. Focus on utilizing or managing the Monthly Surplus Allocation provided in the context. When discussing investments, ONLY use the Retirement/Long-Term portion (₹{{retirement_allocation:.2f}}). Always cite your source if you use general market information.
"""

        # Context for the User Query
        user_context = f"""
User's Financial Snapshot:
- Monthly Income: {format_currency(income)}
- Fixed Expenses Breakdown:
{fixed_breakdown}
- Total Fixed Expenses: {format_currency(total_fixed)}
- Predicted Variable Expense: {format_currency(predicted_variable)}
- Total Predicted Monthly Outflow: {format_currency(total_expected)}
- Current Status: {advice_summary}
- Monthly Surplus: {format_currency(surplus)}
{allocation_advice}

Based on this comprehensive financial data, answer the user's question.
"""

        full_user_query = f"{user_context}\n\nUser Question: {question}"

        # Gemini API Call Payload
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": full_user_query}]
            }],
            "systemInstruction": {
                "parts": [{"text": system_prompt.format(income=income, retirement_allocation=retirement_allocation)}]
            },
            "tools": [{"google_search": {}}]
        }

        # API Call with Exponential Backoff
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    GEMINI_API_URL,
                    headers={'Content-Type': 'application/json'},
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                candidate = result.get('candidates', [{}])[0]
                text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No response generated.')

                sources = []
                grounding_metadata = candidate.get('groundingMetadata', {})
                if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                    sources = [
                        f"[{attr.get('web', {}).get('title', 'Source')}]({attr.get('web', {}).get('uri', '#')})"
                        for attr in grounding_metadata['groundingAttributions']
                        if attr.get('web', {}).get('uri')
                    ]

                source_text = "\n\n**Cited Sources:**\n" + "\n".join(sources) if sources else ""

                markdown_output = f"{predicted_expense_text}\n\n{advice_summary}\n\n{allocation_advice}\n\n💬 Advisor:\n{text}{source_text}"

                return markdown_output

            except requests.exceptions.HTTPError as e:
                if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    raise e
            except Exception as e:
                raise e

        return "❌ Error: Max retries reached or unhandled API error."

    except Exception as e:
        return f"❌ Fatal Error in advisor call: {type(e).__name__}: {e}"

# --- Gradio Interface Wrappers ---
def sanitize_number_input(value):
    """Converts None to 0 for number inputs."""
    return value if value is not None else 0

def handle_initial_report(file, rent_val, emi_val, debt_val, food_val, sub_val, custom_expenses_str, income_val, lang):
    """Generates the initial financial report."""
    fixed_expenses = {
        "Rent": sanitize_number_input(rent_val),
        "EMI/Loan": sanitize_number_input(emi_val),
        "Debt Payments": sanitize_number_input(debt_val),
        "Fixed Food/Groceries": sanitize_number_input(food_val),
        "Subscription": sanitize_number_input(sub_val)
    }
    income_val = sanitize_number_input(income_val)

    custom_expenses = parse_custom_expenses(custom_expenses_str)
    fixed_expenses.update(custom_expenses)

    initial_question = "Give me a quick summary of my financial health and provide actionable next steps on managing the monthly surplus allocation."
    result = ask_advisor(
        question=initial_question,
        fixed_expenses=fixed_expenses,
        income=income_val,
        csv_file=file,
        language=lang
    )

    is_success = "❌" not in result

    # Reset audio and update status
    return [
        result,
        gr.update(visible=is_success),
        gr.update(value=None, visible=False, interactive=False),
        "TTS Status: Ready"
    ]

def handle_chat_response(question, file, rent_val, emi_val, debt_val, food_val, sub_val, custom_expenses_str, income_val, lang):
    """Answers specific user questions."""
    fixed_expenses = {
        "Rent": sanitize_number_input(rent_val),
        "EMI/Loan": sanitize_number_input(emi_val),
        "Debt Payments": sanitize_number_input(debt_val),
        "Fixed Food/Groceries": sanitize_number_input(food_val),
        "Subscription": sanitize_number_input(sub_val)
    }
    income_val = sanitize_number_input(income_val)

    custom_expenses = parse_custom_expenses(custom_expenses_str)
    fixed_expenses.update(custom_expenses)

    result = ask_advisor(question, fixed_expenses, income_val, file, language=lang)

    is_success = "❌" not in result

    # Reset audio and update status
    return [
        result,
        gr.update(visible=is_success),
        gr.update(value=None, visible=False, interactive=False),
        "TTS Status: Ready"
    ]


# --- Gradio Interface ---
with gr.Blocks() as iface:
    # Initialize components using default (English) strings
    md_title = gr.Markdown(UI_STRINGS["English"]["title"])
    md_desc = gr.Markdown(UI_STRINGS["English"]["desc"])

    with gr.Row():
        language_select = gr.Dropdown(
            choices=["English", "Hindi (हिंदी)", "Marathi (मराठी)"],
            label=UI_STRINGS["English"]["lang_label"],
            value="English",
            scale=0
        )
        csv_file = gr.File(
            file_types=[".csv"],
            label=UI_STRINGS["English"]["csv_label"],
            scale=1
        )

    with gr.Row():
        income = gr.Number(label=UI_STRINGS["English"]["income_label"], value=0, minimum=0)

    with gr.Row():
        rent = gr.Number(label=UI_STRINGS["English"]["rent_label"], value=0, minimum=0)
        emi = gr.Number(label=UI_STRINGS["English"]["emi_label"], value=0, minimum=0)
        debt = gr.Number(label=UI_STRINGS["English"]["debt_label"], value=0, minimum=0)
        food = gr.Number(label=UI_STRINGS["English"]["food_label"], value=0, minimum=0)
        subscription = gr.Number(label=UI_STRINGS["English"]["subscription_label"], value=0, minimum=0)

    custom_placeholder_en = "e.g., Gym 1500 or PetCare:500 (Separate items with space, comma, or newline)"
    with gr.Row():
        custom_expenses = gr.Textbox(
            label=UI_STRINGS["English"]["custom_label"],
            placeholder=custom_placeholder_en,
            lines=2
        )

    upload_btn = gr.Button(UI_STRINGS["English"]["upload_btn"])

    with gr.Column():
        output_text = gr.Markdown(label=UI_STRINGS["English"]["output_label"])
        report_speak_btn = gr.Button(UI_STRINGS["English"]["speak_btn"], visible=False, size="sm")

    md_chat_title = gr.Markdown(UI_STRINGS["English"]["chat_title"])

    with gr.Row():
        user_question = gr.Textbox(
            label=UI_STRINGS["English"]["question_label"],
            placeholder=UI_STRINGS["English"]["question_placeholder"],
            scale=3
        )
        submit_btn = gr.Button(UI_STRINGS["English"]["submit_btn"], scale=1)

    with gr.Column():
        chat_output = gr.Markdown(label=UI_STRINGS["English"]["chat_output_label"])
        chat_speak_btn = gr.Button(UI_STRINGS["English"]["speak_btn"], visible=False, size="sm")

    # Audio and Status Components
    audio_output = gr.Audio(visible=False, interactive=False)
    tts_status = gr.Textbox(UI_STRINGS["English"]["tts_status"] + " Ready", visible=True, show_label=False)

    # UI Language Update Handler
    def update_ui_language(lang):
        """Updates all Gradio UI component texts based on the selected language."""
        strings = UI_STRINGS[lang]

        if lang == "English":
            new_custom_placeholder = custom_placeholder_en
        else:
            base_example = strings["custom_label"].split('(')[1].replace(')', '')
            new_custom_placeholder = f"{base_example} (Separate items with space, comma, or newline)"

        return [
            gr.update(value=strings["title"]),
            gr.update(value=strings["desc"]),
            gr.update(label=strings["lang_label"]),
            gr.update(label=strings["csv_label"]),
            gr.update(label=strings["income_label"]),
            gr.update(label=strings["rent_label"]),
            gr.update(label=strings["emi_label"]),
            gr.update(label=strings["debt_label"]),
            gr.update(label=strings["food_label"]),
            gr.update(label=strings["subscription_label"]),
            gr.update(label=strings["custom_label"], placeholder=new_custom_placeholder),
            gr.update(value=strings["upload_btn"]),
            gr.update(label=strings["output_label"]),
            gr.update(value=strings["chat_title"]),
            gr.update(label=strings["question_label"], placeholder=strings["question_placeholder"]),
            gr.update(value=strings["submit_btn"]),
            gr.update(label=strings["chat_output_label"]),
            gr.update(value=strings["speak_btn"]),
            gr.update(value=strings["speak_btn"]),
            gr.update(value=strings["tts_status"] + " Ready")
        ]

    # Gradio Component List for Update Target
    all_components = [
        md_title, md_desc, language_select, csv_file, income, rent, emi, debt, food,
        subscription, custom_expenses, upload_btn, output_text, md_chat_title, user_question,
        submit_btn, chat_output, report_speak_btn, chat_speak_btn, tts_status
    ]

    language_select.change(
        fn=update_ui_language,
        inputs=[language_select],
        outputs=all_components
    )


    # Initial Report Generation
    upload_btn.click(
        fn=handle_initial_report,
        inputs=[csv_file, rent, emi, debt, food, subscription, custom_expenses, income, language_select],
        outputs=[output_text, report_speak_btn, audio_output, tts_status]
    )

    # Chat Response Generation
    submit_btn.click(
        fn=handle_chat_response,
        inputs=[user_question, csv_file, rent, emi, debt, food, subscription, custom_expenses, income, language_select],
        outputs=[chat_output, chat_speak_btn, audio_output, tts_status]
    )

    # TTS for Initial Report
    report_speak_btn.click(
        fn=speak_response,
        inputs=[output_text, language_select],
        outputs=[audio_output, tts_status]
    )

    # TTS for Chat Response
    chat_speak_btn.click(
        fn=speak_response,
        inputs=[chat_output, language_select],
        outputs=[audio_output, tts_status]
    )


# Launch app
if __name__ == "__main__":
    iface.launch()