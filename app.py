"""Swasth Sahayak - private, offline AI health document assistant.
Run: streamlit run app.py   (needs Ollama running locally)"""
import numpy as np
import streamlit as st
import ollama
from PIL import Image

from tools import (check_drugs, check_labs, medicine_table, facts_text,
                   report_terms, report_conclusion, report_facts)

st.set_page_config(page_title="Swasth Sahayak", page_icon="🩺", layout="wide")

LANGS = {
    "English": "English", "हिन्दी (Hindi)": "Hindi", "मराठी (Marathi)": "Marathi",
    "বাংলা (Bengali)": "Bengali", "தமிழ் (Tamil)": "Tamil", "తెలుగు (Telugu)": "Telugu",
    "ಕನ್ನಡ (Kannada)": "Kannada", "മലയാളം (Malayalam)": "Malayalam",
    "ગુજરાતી (Gujarati)": "Gujarati", "ਪੰਜਾਬੀ (Punjabi)": "Punjabi", "ଓଡ଼ିଆ (Odia)": "Odia",
}

# OCR: (engine, language codes). EasyOCR handles these scripts; Tesseract covers the rest.
OCR_OPTS = {
    "English only": ("easy", ["en"]),
    "Devanagari - Hindi/Marathi (+English)": ("easy", ["en", "hi"]),
    "Bengali (+English)": ("easy", ["en", "bn"]),
    "Tamil (+English)": ("easy", ["en", "ta"]),
    "Telugu (+English)": ("easy", ["en", "te"]),
    "Kannada (+English)": ("easy", ["en", "kn"]),
    "Malayalam - Tesseract": ("tess", "eng+mal"),
    "Gujarati - Tesseract": ("tess", "eng+guj"),
    "Punjabi - Tesseract": ("tess", "eng+pan"),
    "Odia - Tesseract": ("tess", "eng+ori"),
}
LANG_TO_OCR = {
    "Hindi": "Devanagari - Hindi/Marathi (+English)", "Marathi": "Devanagari - Hindi/Marathi (+English)",
    "Bengali": "Bengali (+English)", "Tamil": "Tamil (+English)", "Telugu": "Telugu (+English)",
    "Kannada": "Kannada (+English)", "Malayalam": "Malayalam - Tesseract",
    "Gujarati": "Gujarati - Tesseract", "Punjabi": "Punjabi - Tesseract", "Odia": "Odia - Tesseract",
}
SAMPLE = """Dr. A. Sharma, MBBS  | Patient: Ramesh, 58 yrs
Rx:
1. Tab Ecosprin 75 mg  1-0-0  after food
2. Tab Brufen 400 mg  1-0-1  after food x 5 days
3. Tab Warfarin 5 mg  0-0-1
4. Tab Dolo 650  SOS for fever
5. Tab Crocin 500  SOS
Lab: Hemoglobin 9.8 g/dL, Fasting glucose 142 mg/dL, HbA1c 7.4 %, Creatinine 1.1 mg/dL"""

SYSTEM = """You are Swasth Sahayak, a kind helper who explains medical papers to ordinary people in India, including people with little schooling.
Imagine you are talking to a 60-year-old villager sitting in front of you.

STYLE RULES (most important):
- Reply ONLY in {lang}.
- Use SHORT sentences (about 12 words or fewer) and everyday spoken words. No medical jargon.
- If a medical word cannot be avoided, say it in simple words first and put the medical word once in brackets.
- Never use words like: etiology, echotexture, hypoechoic, necrosis, lymphadenopathy, bilateral, discrete, parameters, differential.
- Do NOT copy sentences from the facts. Say them again more simply.
- BAD: "Cervical lymphadenopathy with infective etiology."
  GOOD: "The glands in your neck are swollen. The doctor thinks it is most likely from an infection (germs)."

ALWAYS USE THIS LAYOUT (translate the headings into {lang}):
## 1. In short
2 or 3 simple sentences: the main point of this paper.
## 2. ✅ What is fine
One short bullet list of the normal things.
## 3. ⚠️ What needs attention
One bullet for each problem: what it is, in one simple sentence. Write "Nothing" if there is none.
## 4. 💊 Medicines
Only if there are medicines. One bullet each: the name, what it is for, when to take it.
## 5. What to do next
2 or 3 bullets, for example: show this paper to your doctor, ask the doctor questions, do not change medicines on your own.

CONTENT RULES:
- Use ONLY the information in VERIFIED FACTS, SCAN REPORT TERMS, RADIOLOGIST CONCLUSION and SAFETY CHECK FINDINGS. Never add medical facts, uses, doses or side effects from your own memory.
- If a medicine is marked NO VERIFIED INFORMATION, say you do not have information about it and the user should ask the pharmacist. Do not guess.
- Mention every safety warning clearly.
- Never say a condition is present, absent, serious, mild or 'nothing to worry about' unless the paper itself says so. Do not name any disease the paper does not name.
- Never diagnose, never change doses, never tell the user to start or stop a medicine.
- Keep medicine names exactly as written in the document.
- Say that the doctor will decide the next tests and treatment.
- End with one line: please show this to your doctor."""


@st.cache_resource(show_spinner="Loading OCR model (first run only)...")
def get_reader(langs: tuple):
    import easyocr
    return easyocr.Reader(list(langs), gpu=False)


def run_ocr(img, opt):
    engine, code = OCR_OPTS[opt]
    if engine == "easy":
        return "\n".join(get_reader(tuple(code)).readtext(np.array(img), detail=0))
    import pytesseract  # needs system Tesseract + language packs (see README)
    return pytesseract.image_to_string(img, lang=code)


def local_models():
    try:
        res = ollama.list()
        models = res["models"]
        return [m.get("model") or m.get("name") for m in models] if models else []
    except Exception:
        return []


def stream_llm(model, messages):
    for chunk in ollama.chat(model=model, messages=messages, stream=True, options={"temperature": 0.2}):
        yield chunk["message"]["content"]


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚙️ Settings")
    models = local_models()
    if not models:
        st.error("Ollama not reachable or no model installed.\n\nRun: `ollama pull gemma3:12b`")
    model = st.selectbox("Local model", models or ["gemma3:12b"])
    lang_label = st.selectbox("Explain in", list(LANGS))
    lang = LANGS[lang_label]
    ocr_default = LANG_TO_OCR.get(lang, "English only")
    ocr_opt = st.selectbox("Document script (for OCR)", list(OCR_OPTS),
                           index=list(OCR_OPTS).index(ocr_default), key=f"ocr_{lang}")
    two_step = st.checkbox("🎯 Better quality for regional languages (2-step)", value=lang != "English",
                           help="Explains in English first, then translates. Slower but more accurate on small models.")
    st.success("🔒 100% local. No data leaves this device.")
    st.caption("Tip: turn off WiFi to prove it.")
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

st.title("🩺 Swasth Sahayak - Private Health Document Assistant")
st.caption("Upload a prescription or lab report. Get a simple explanation and safety alerts, fully offline.")

# ---------------- Input ----------------
col1, col2 = st.columns(2)
with col1:
    up = st.file_uploader("📷 Prescription / report photo", type=["png", "jpg", "jpeg"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, use_container_width=True)
        if st.button("🔍 Read text (OCR)"):
            with st.spinner("Reading..."):
                try:
                    st.session_state["text"] = run_ocr(img, ocr_opt)
                except Exception as e:
                    st.error(f"OCR failed: {e}")
    if st.button("Load sample prescription"):
        st.session_state["text"] = SAMPLE

with col2:
    text = st.text_area("Extracted text (edit to fix OCR mistakes)",
                        value=st.session_state.get("text", ""), height=300)

# ---------------- Analysis ----------------
if st.button("✨ Explain & check safety", type="primary", disabled=not text.strip()):
    drugs, warnings = check_drugs(text)
    labs = check_labs(text)
    med_rows, unknown_meds = medicine_table(text)
    terms = report_terms(text)
    concl = report_conclusion(text)

    if warnings:
        st.subheader("🛡️ Important safety alerts")
        for w in warnings:
            (st.error if "[HIGH]" in w else st.warning)(w)
    elif med_rows:
        st.success("No dangerous medicine combination found in our list (this is not a guarantee).")
    if unknown_meds:
        st.info("We have no information about: " + ", ".join(unknown_meds) + ". Please ask your pharmacist.")

    findings = "\n".join(warnings) or "None found."
    findings += "\nLab results: " + ("; ".join(f'{l["test"]} {l["value"]} {l["unit"]} is {l["status"]} - {l["meaning"]}' for l in labs) or "none")
    messages = [
        {"role": "system", "content": SYSTEM.format(lang=lang)},
        {"role": "user", "content": f"DOCUMENT TEXT:\n{text}\n\nVERIFIED FACTS (medicines):\n{facts_text(med_rows, unknown_meds)}\n\n"
                                    f"SCAN REPORT TERMS:\n{report_facts(terms, concl)}\n\n"
                                    f"RADIOLOGIST CONCLUSION (copy exactly):\n{concl or 'none'}\n\n"
                                    f"SAFETY CHECK FINDINGS:\n{findings}\n\n"
                                    f"Explain this document in {lang}, in very simple words, using the layout from your rules."},
    ]
    st.subheader("💬 Simple explanation")
    try:
        if two_step and lang != "English":
            en_msgs = [{"role": "system", "content": SYSTEM.format(lang="English")}, messages[1]]
            with st.spinner("Step 1/2: understanding the document..."):
                english = "".join(stream_llm(model, en_msgs))
            tr_msgs = [{"role": "user", "content":
                        f"Translate the text below into {lang}. Use simple everyday words a villager understands. "
                        "Keep medicine names, numbers and units in English letters. Keep every warning. "
                        "Keep the same headings, bullets and emojis. Write the way people speak at home, "
                        "not bookish or formal words. Keep sentences short. "
                        f"Output only the {lang} translation.\n\n{english}"}]
            answer = st.write_stream(stream_llm(model, tr_msgs))
        else:
            answer = st.write_stream(stream_llm(model, messages))
        st.session_state["chat"] = messages + [{"role": "assistant", "content": answer}]
    except Exception as e:
        st.error(f"Could not reach the local model: {e}")

    # Technical details, collapsed so the simple explanation stays in front
    if med_rows:
        with st.expander("💊 Medicine details (verified)"):
            st.table(med_rows)
    if labs:
        with st.expander("🧪 Lab result details"):
            st.table([{"Test": l["test"], "Value": f'{l["value"]} {l["unit"]}', "Normal range": l["range"],
                       "Status": l["status"], "Meaning": l["meaning"]} for l in labs])
    if terms:
        with st.expander("📄 Meaning of medical words in your report"):
            if concl:
                st.info(f"Doctor's conclusion, as written: {concl}")
            st.table(terms)

# ---------------- Follow-up chat ----------------
if "chat" in st.session_state:
    st.divider()
    st.subheader("❓ Ask a follow-up")
    for m in st.session_state["chat"][2:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    if q := st.chat_input(f"Ask in {lang}..."):
        st.session_state["chat"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            a = st.write_stream(stream_llm(model, st.session_state["chat"]))
        st.session_state["chat"].append({"role": "assistant", "content": a})

st.divider()
st.caption("⚠️ Educational aid only. Not a substitute for a doctor or pharmacist. Always confirm with a professional.")
