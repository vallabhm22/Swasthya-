"""Deterministic 'agent tools': drug detection, interaction check, lab-range flags.
These run locally with no model, so safety findings never depend on LLM accuracy."""
import re
import difflib

# Brand/generic name -> generic ingredient(s). Extend freely.
DRUGS = {
    "dolo": ["paracetamol"], "crocin": ["paracetamol"], "calpol": ["paracetamol"],
    "paracetamol": ["paracetamol"], "acetaminophen": ["paracetamol"],
    "combiflam": ["ibuprofen", "paracetamol"], "brufen": ["ibuprofen"], "ibuprofen": ["ibuprofen"],
    "disprin": ["aspirin"], "ecosprin": ["aspirin"], "aspirin": ["aspirin"],
    "voveran": ["diclofenac"], "diclofenac": ["diclofenac"],
    "warfarin": ["warfarin"], "clopidogrel": ["clopidogrel"], "plavix": ["clopidogrel"],
    "omeprazole": ["omeprazole"], "omez": ["omeprazole"],
    "pantoprazole": ["pantoprazole"], "pantop": ["pantoprazole"], "pan 40": ["pantoprazole"],
    "metformin": ["metformin"], "glycomet": ["metformin"],
    "enalapril": ["ace_inhibitor"], "ramipril": ["ace_inhibitor"], "lisinopril": ["ace_inhibitor"],
    "spironolactone": ["spironolactone"], "aldactone": ["spironolactone"],
    "tramadol": ["tramadol"], "sertraline": ["sertraline"],
    "simvastatin": ["simvastatin"], "clarithromycin": ["clarithromycin"],
    "sildenafil": ["sildenafil"], "nitroglycerin": ["nitrate"], "isosorbide": ["nitrate"],
    "amoxicillin": ["amoxicillin"], "azithromycin": ["azithromycin"],
    "atorvastatin": ["atorvastatin"], "amlodipine": ["amlodipine"],
}

INTERACTIONS = [
    ({"warfarin", "aspirin"}, "HIGH", "Warfarin + Aspirin: both thin the blood. Together they can cause serious bleeding."),
    ({"warfarin", "ibuprofen"}, "HIGH", "Warfarin + Ibuprofen: ibuprofen increases the bleeding risk of warfarin. This can be serious."),
    ({"warfarin", "diclofenac"}, "HIGH", "Warfarin + Diclofenac: diclofenac increases the bleeding risk of warfarin. This can be serious."),
    ({"aspirin", "ibuprofen"}, "MEDIUM", "Aspirin + Ibuprofen: together they can hurt the stomach and cause bleeding. Ibuprofen can also weaken the heart protection of aspirin."),
    ({"clopidogrel", "omeprazole"}, "MEDIUM", "Clopidogrel + Omeprazole: omeprazole can make clopidogrel work less well."),
    ({"ace_inhibitor", "spironolactone"}, "MEDIUM", "ACE inhibitor + Spironolactone: together they can raise the potassium in the blood too much."),
    ({"tramadol", "sertraline"}, "HIGH", "Tramadol + Sertraline: together they can cause fits (seizures) and a dangerous reaction called serotonin syndrome."),
    ({"simvastatin", "clarithromycin"}, "HIGH", "Simvastatin + Clarithromycin: together they can badly damage the muscles."),
    ({"sildenafil", "nitrate"}, "HIGH", "Sildenafil + Nitrates: together they can drop the blood pressure dangerously low."),
]

# name -> (aliases regex, low, high, unit)
LABS = {
    "Hemoglobin": (r"hemoglobin|haemoglobin|hb", 12.0, 17.0, "g/dL"),
    "WBC count": (r"wbc|total leu[ck]ocyte count|tlc", 4000, 11000, "/cumm"),
    "Platelets": (r"platelets?(?: count)?", 150000, 450000, "/cumm"),
    "Fasting glucose": (r"fasting (?:blood )?(?:glucose|sugar)|fbs|glucose", 70, 100, "mg/dL"),
    "HbA1c": (r"hba1c|glycosylated h[a]?emoglobin", 4.0, 5.6, "%"),
    "Total cholesterol": (r"total cholesterol|cholesterol", 0, 200, "mg/dL"),
    "LDL": (r"ldl(?: cholesterol)?", 0, 100, "mg/dL"),
    "HDL": (r"hdl(?: cholesterol)?", 40, 100, "mg/dL"),
    "Triglycerides": (r"triglycerides?", 0, 150, "mg/dL"),
    "Creatinine": (r"creatinine", 0.6, 1.3, "mg/dL"),
    "TSH": (r"tsh", 0.4, 4.0, "uIU/mL"),
}



# generic -> (what it is used for, key caution). Written in plain language; the LLM only rephrases these.
DRUG_INFO = {
    "paracetamol": ("takes away mild pain and fever", "too much harms the liver; never take more than written, and do not take two paracetamol medicines together"),
    "ibuprofen": ("takes away pain, swelling and fever", "take after food; it can upset the stomach and harm the kidneys"),
    "aspirin": ("in a small dose (like 75 mg) it keeps the blood thin so clots do not form and cause a heart attack or stroke; in a bigger dose it takes away pain and fever", "can upset the stomach and cause bleeding"),
    "diclofenac": ("takes away pain and swelling", "take after food; it can upset the stomach and harm the kidneys"),
    "warfarin": ("keeps the blood thin so dangerous clots do not form", "can cause bleeding; needs regular blood tests (INR); many medicines react with it"),
    "clopidogrel": ("stops blood from clotting; given for heart disease or after a stroke", "can cause bleeding; do not stop it without asking the doctor"),
    "omeprazole": ("reduces acid in the stomach, for acidity and ulcers", "usually taken before food"),
    "pantoprazole": ("reduces acid in the stomach, for acidity and ulcers", "usually taken before food"),
    "metformin": ("lowers blood sugar in type 2 diabetes", "take with or after food; it may upset the stomach"),
    "ace_inhibitor": ("lowers blood pressure and protects the heart and kidneys", "can cause a dry cough"),
    "spironolactone": ("a water tablet that removes extra water from the body; used for heart problems, swelling or high blood pressure", "can raise potassium in the blood"),
    "tramadol": ("a strong painkiller", "can cause sleepiness and can become a habit; do not drink alcohol"),
    "sertraline": ("treats low mood (depression) and anxiety", "takes a few weeks to work; do not stop it suddenly"),
    "simvastatin": ("lowers cholesterol (fat in the blood)", "tell the doctor about muscle pain that has no reason"),
    "atorvastatin": ("lowers cholesterol (fat in the blood)", "tell the doctor about muscle pain that has no reason"),
    "clarithromycin": ("an antibiotic that kills germs in infections", "finish the full course even if you feel better"),
    "azithromycin": ("an antibiotic that kills germs in infections", "finish the full course even if you feel better"),
    "amoxicillin": ("an antibiotic that kills germs in infections", "finish the full course even if you feel better; tell the doctor if you are allergic to penicillin"),
    "sildenafil": ("used for erectile dysfunction, or for high blood pressure in the lungs, depending on the dose", "never take it with nitrate heart medicines"),
    "nitrate": ("relieves chest pain from heart problems", "can cause headache and low blood pressure"),
    "amlodipine": ("lowers blood pressure", "can cause swollen ankles"),
}

LAB_MEANING = {
    "Hemoglobin": {"LOW": "Your blood is low (anaemia). You may feel weak or tired. The doctor should find the reason, often lack of iron.",
                   "HIGH": "Higher than usual. It can happen when the body lacks water. Ask the doctor."},
    "WBC count": {"LOW": "Fewer germ-fighting cells than usual. Ask the doctor.", "HIGH": "Often seen when the body is fighting an infection."},
    "Platelets": {"LOW": "Low platelets can cause easy bruising or bleeding. Show the doctor soon.", "HIGH": "Higher than usual. It can happen after an infection. Ask the doctor."},
    "Fasting glucose": {"HIGH": "Your sugar is higher than normal on an empty stomach. This can mean pre-diabetes or diabetes. Only a doctor can confirm.", "LOW": "Your sugar is low. You can feel shaky, sweaty or dizzy. This needs attention."},
    "HbA1c": {"HIGH": "This shows your sugar level over the last 2-3 months. It is higher than normal. The doctor will decide if it is diabetes.", "LOW": "Unusually low. Ask the doctor."},
    "Total cholesterol": {"HIGH": "High cholesterol (fat in the blood) can harm the heart over time."},
    "LDL": {"HIGH": "LDL is the 'bad' fat in the blood. Yours is high, which is not good for the heart."},
    "HDL": {"LOW": "HDL is the 'good' fat in the blood. Yours is low.", "HIGH": "A high HDL is usually fine."},
    "Triglycerides": {"HIGH": "The fat in your blood is high. Sweets, oily food and alcohol can raise it."},
    "Creatinine": {"HIGH": "The kidneys may not be cleaning the blood well. The doctor should check.", "LOW": "Usually not a problem."},
    "TSH": {"HIGH": "Can mean the thyroid gland is working too slowly.", "LOW": "Can mean the thyroid gland is working too fast."},
}

SINGLE_WORD = [k for k in DRUGS if " " not in k]
FORM_RE = re.compile(r"^\s*(?:\d+\s*[.)]\s*)?(?:tab|tablet|cap|capsule|syp|syrup|inj|injection|oint|drops?)\b\.?\s+([A-Za-z][\w-]*)", re.I)


def match_line(line: str):
    """Known drug names in one line. Exact match first, then fuzzy match to survive OCR typos."""
    low = line.lower()
    hits = [n for n in DRUGS if re.search(rf"\b{re.escape(n)}\b", low)]
    if not hits:
        for tok in re.findall(r"[a-z]{5,}", low):
            m = difflib.get_close_matches(tok, SINGLE_WORD, n=1, cutoff=0.84)
            if m:
                hits.append(m[0])
    return hits


def parse_schedule(line: str):
    """Turn 1-0-1, SOS, BD, 'after food', 'x 5 days' into plain words."""
    low = line.lower()
    parts = []
    m = re.search(r"\b([0-3])\s*-\s*([0-3])\s*-\s*([0-3])(?:\s*-\s*([0-3]))?\b", low)
    if m:
        slots = []
        for name, v in zip(["morning", "afternoon", "night", "bedtime"], m.groups()):
            if v and v != "0":
                slots.append(name if v == "1" else f"{v} tablets in the {name}")
        parts.append(" + ".join(slots) if slots else "no fixed time")
    if re.search(r"\b(sos|prn)\b", low): parts.append("only when needed (SOS)")
    if re.search(r"\bod\b", low): parts.append("once a day")
    if re.search(r"\b(bd|bid)\b", low): parts.append("twice a day")
    if re.search(r"\b(tds|tid)\b", low): parts.append("three times a day")
    if re.search(r"\bhs\b", low): parts.append("at bedtime")
    for k, v in [("after food", "after food"), ("before food", "before food"),
                 ("empty stomach", "on an empty stomach"), ("with food", "with food")]:
        if k in low: parts.append(v)
    d = re.search(r"(?:x|for)\s*(\d+)\s*(days?|weeks?)", low)
    if d: parts.append(f"for {d.group(1)} {d.group(2)}")
    s = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", low)
    strength = f"{s.group(1)} {s.group(2)}" if s else ""
    return (", ".join(parts) or "not clearly written - confirm with the pharmacist"), strength


def find_drugs(text: str):
    """Return {generic: [brand/names found]}."""
    found = {}
    for line in text.splitlines():
        for name in match_line(line):
            for g in DRUGS[name]:
                found.setdefault(g, []).append(name)
    return found


def medicine_table(text: str):
    """Verified rows (one per known medicine line) and a list of unverified medicine names."""
    rows, unknown = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        hits = list(dict.fromkeys(match_line(line)))
        if not hits:
            m = FORM_RE.match(line)
            if m:
                unknown.append(m.group(1))
            continue
        schedule, strength = parse_schedule(line) if len(hits) == 1 else ("see the document line", "")
        for h in hits:
            gens = DRUGS[h]
            rows.append({
                "Medicine": h.title() + (f" {strength}" if strength else ""),
                "Contains": ", ".join(g.replace("_", " ") for g in gens),
                "Used for": "; ".join(DRUG_INFO[g][0] for g in gens),
                "How to take": schedule,
                "Caution": "; ".join(DRUG_INFO[g][1] for g in gens),
            })
    return rows, list(dict.fromkeys(unknown))


def facts_text(rows, unknown):
    """Ground-truth block handed to the LLM."""
    out = []
    for r in rows:
        out.append(f"- {r['Medicine']} (contains {r['Contains']}) | used for: {r['Used for']} | "
                   f"how to take: {r['How to take']} | caution: {r['Caution']}")
    for u in unknown:
        out.append(f"- {u}: NO VERIFIED INFORMATION AVAILABLE. Do not describe it; tell the user to ask the pharmacist.")
    return "\n".join(out) or "No medicines recognised."


def check_drugs(text: str):
    """Return (drugs_found, warnings)."""
    found = find_drugs(text)
    warnings = []
    for pair, level, msg in INTERACTIONS:
        if pair <= set(found):
            warnings.append(f"[{level}] {msg}")
    for g, names in found.items():
        if len(set(names)) > 1 and g != "ace_inhibitor":
            warnings.append(f"[MEDIUM] Same medicine twice: {', '.join(sorted(set(names)))} all contain {g}. Taking them together can mean too much of the same medicine (overdose).")
    return found, warnings


def check_labs(text: str):
    """Return list of dicts: test, value, unit, status, range, meaning."""
    results = []
    low = text.lower()
    for test, (alias, lo, hi, unit) in LABS.items():
        val = None
        for m in re.finditer(rf"\b(?:{alias})\b[^\d\n]{{0,30}}?(\d+(?:[.,]\d+)?)", low):
            prefix = low[max(0, m.start() - 14):m.start()]
            if test == "Hemoglobin" and prefix.endswith("glycosylated "):
                continue
            if test == "Total cholesterol" and re.search(r"(ldl|hdl|vldl)[ -]?$", prefix):
                continue
            val = float(m.group(1).replace(",", "."))
            break
        if val is None:
            continue
        if test == "Platelets":  # e.g. "2.5 lakh" or "250 x10^3"
            val = val * 100000 if val < 10 else val * 1000 if val < 1000 else val
        if test == "WBC count" and val < 100:
            val *= 1000
        status = "LOW" if val < lo else "HIGH" if val > hi else "NORMAL"
        meaning = LAB_MEANING.get(test, {}).get(status, "Within the usual adult range." if status == "NORMAL" else "Outside the usual range. Ask the doctor.")
        results.append({"test": test, "value": val, "unit": unit, "status": status,
                        "range": f"{lo}-{hi}", "meaning": meaning})
    return results


# ---------------- Scan / imaging report glossary (verified plain-language meanings) ----------------
REPORT_TERMS = [
    (r"\b(?:sonography|usg|ultrasound)\b", "Sonography / USG", "A scan that uses sound waves to look inside the body. It is painless and has no radiation."),
    (r"\bthyroid\b", "Thyroid", "A small gland at the front of the neck. It controls how your body uses energy."),
    (r"homogen(?:e)?ous", "Homogenous echotexture", "The inside looks even and healthy on the scan."),
    (r"hypo-?echoic", "Hypoechoic area", "A darker patch on the scan. Read the report line to see if one was found."),
    (r"\bcysts?\b|\bcystic\b", "Cyst / cystic", "A small bag filled with fluid."),
    (r"smooth margins?", "Smooth margins", "The edges look smooth and regular, which is normal."),
    (r"carotid", "Carotid arteries", "The big blood pipes in the neck that carry blood to the brain."),
    (r"jugular|\b[il1]jv\b", "Jugular vein / IJV", "The big vein in the neck that brings blood back from the head. Doctors use it as a landmark to say where a gland is."),
    (r"submandibular", "Submandibular", "The area under the jaw. It has glands that make saliva (spit)."),
    (r"parotid", "Parotid glands", "Glands near the ears and cheeks that make saliva (spit)."),
    (r"\bbilateral\b", "Bilateral", "On both sides (left and right)."),
    (r"lymph nodes?", "Lymph nodes", "Small bean-sized glands in the neck, armpit and groin. They help fight germs. They swell up when the body fights an infection, and sometimes for other reasons."),
    (r"\bdiscrete\b", "Discrete", "Separate from each other, not joined into one lump."),
    (r"non[- ]?necrotic", "Non-necrotic", "No dead or damaged tissue inside the glands."),
    (r"\bnecrosis\b", "Necrosis", "Dead tissue inside a gland. Doctors check for it because it can point to some serious infections or other problems."),
    (r"level\s*(?:i{1,3}|iv|v|[1-6])\b", "Level II / III / IV", "Doctors divide the neck into zones numbered I to VI. Levels II to IV are the middle part of the side of the neck."),
    (r"\d+\s*x\s*\d+\s*mm", "Size (for example 21 x 8 mm)", "Size in millimetres (10 mm = 1 cm, about the width of a fingernail). The first number is the longer side, the second is the shorter side."),
    (r"short axis", "Short axis", "The shorter side of a gland. Doctors mostly use it to decide if a gland is really swollen."),
    (r"sub-?centimet", "Subcentimeter", "Smaller than 1 cm."),
    (r"lymphadenopathy", "Lymphadenopathy", "Swollen glands (lymph nodes). 'Cervical' means in the neck."),
    (r"[ae]?etiology", "Etiology", "The cause."),
    (r"\binfective\b", "Infective", "Caused by germs (an infection)."),
    (r"rule out", "To rule out", "The doctor wants to check if this is the reason. It is something to check, not a confirmed finding."),
    (r"\bkoch'?s?\b", "Koch's", "Tuberculosis (TB), a germ infection. The report only says it should be checked with more tests. It does not say you have TB."),
    (r"fatty liver|steatosis", "Fatty liver", "Extra fat in the liver. Grade I means a little, grade II medium, grade III a lot."),
    (r"hepatomegaly", "Hepatomegaly", "The liver is bigger than normal."),
    (r"splenomegaly", "Splenomegaly", "The spleen is bigger than normal."),
    (r"calcul(?:us|i)|\bstones?\b", "Calculus (stone)", "A hard stone, for example in the kidney or gall bladder."),
    (r"hydronephrosis", "Hydronephrosis", "The kidney is swollen because urine is not flowing out well."),
    (r"ascites|free fluid", "Free fluid / ascites", "Water collected inside the belly."),
    (r"fibroid", "Fibroid", "A lump in the womb that is not cancer."),
    (r"\bpolyp\b", "Polyp", "A small lump growing on the inside lining of an organ."),
]


def report_terms(text: str):
    """Verified meanings of scan-report terms found in the text, in order of appearance."""
    low = text.lower()
    found = []
    for pattern, label, meaning in REPORT_TERMS:
        m = re.search(pattern, low)
        if m:
            found.append((m.start(), label, meaning))
    found.sort()
    return [{"Term": l, "What it means": mn} for _, l, mn in found]


def report_conclusion(text: str):
    """The radiologist's conclusion/impression, copied as written."""
    m = re.search(r"(?:conclusion|impression)\s*:?\s*-*\s*(.+?)(?:\n[\s|]*(?:dr\.|md\b|consultant\b)|\Z)", text, re.I | re.S)
    return " ".join(m.group(1).split()) if m else ""


def report_facts(terms, conclusion):
    out = [f"- {t['Term']}: {t['What it means']}" for t in terms]
    return "\n".join(out) or "None."
