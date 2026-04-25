"""
llm_engine.py (v4 — GEMINI REAL-TIME)
---------------------------------------
Uses Google Gemini API for real-time misinformation analysis.
Falls back to heuristic analyzer if API is unavailable.

Setup:
  pip install google-generativeai
  export GEMINI_API_KEY="AIzaSy..."

FINAL_SCORE = (ML * 0.30) + (Gemini * 0.50) + (EVIDENCE * 0.20)
"""

import os
import re
import json
import hashlib
from typing import Optional

try:
    import google.generativeai as genai
    _GEMINI_SDK_OK = True
except ImportError:
    _GEMINI_SDK_OK = False

_llm_cache: dict = {}

def _cache_key(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

GEMINI_SYSTEM_PROMPT = """You are VeriLens AI, an expert misinformation analyst and fact-checker.

Your job is to analyze news headlines or claims and determine if they are REAL or FAKE.

Rules:
- Use your real-world knowledge to fact-check claims
- Identify sensational language, conspiracy theories, logical fallacies
- Be decisive — avoid UNCERTAIN unless genuinely ambiguous
- For universal facts (scientific consensus, biology, history), be very confident
- For recent news events, analyze language patterns carefully

Respond ONLY with a valid JSON object. No markdown, no extra text, no backticks.

JSON schema:
{
  "verdict": "FAKE" | "REAL" | "UNCERTAIN",
  "confidence": <float 0.0-1.0>,
  "real_score": <float 0.0-1.0, probability it is real>,
  "reasoning": "<2-3 sentence explanation for general audience>",
  "red_flags": ["<flag1>", "<flag2>"],
  "credibility_signals": ["<signal1>"],
  "claim_type": "factual" | "opinion" | "satire" | "conspiracy" | "misleading",
  "recommendation": "<one sentence advice for the reader>"
}"""

def _call_gemini(text: str) -> Optional[dict]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or not _GEMINI_SDK_OK:
        return None

    ck = _cache_key(text)
    if ck in _llm_cache and _llm_cache[ck].get("_source") == "gemini":
        return _llm_cache[ck]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=GEMINI_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=600,
            )
        )
        prompt = f"Analyze this claim for misinformation:\n\n{text[:3000]}"
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        data = json.loads(raw)
        if "real_score" not in data:
            data["real_score"] = (1.0 - data.get("confidence", 0.8)
                if data.get("verdict") == "FAKE"
                else data.get("confidence", 0.8))
        data["_source"] = "gemini"
        _llm_cache[ck] = data
        print(f"[Gemini] verdict={data['verdict']} conf={data['confidence']:.2f}")
        return data
    except json.JSONDecodeError as e:
        print(f"[Gemini] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"[Gemini] API call failed: {e}")
        return None

import re as _re

KNOWLEDGE_BASE = [
    (r"\b(boy|man|male|men)\b.{0,30}\b(give birth|get pregnant|carry baby)\b", False, 0.99, "Biological impossibility: males cannot give birth."),
    (r"\bgive birth\b.{0,30}\b(boy|male|man|men)\b", False, 0.99, "Biological impossibility: males cannot give birth."),
    (r"\bmale.{0,20}(pregnant|pregnancy)\b", False, 0.97, "Males cannot naturally become pregnant."),
    (r"\bearth\b.{0,30}\bflat\b", False, 0.99, "Earth is an oblate spheroid, not flat."),
    (r"\bflat earth\b", False, 0.99, "Flat Earth is thoroughly debunked."),
    (r"\bmoon landing\b.{0,40}\b(fake|faked|staged|hoax)\b", False, 0.99, "Apollo 11 moon landing is one of the most documented events in history."),
    (r"\bvaccine\w*\b.{0,40}\b(cause|causes)\b.{0,20}\bautism\b", False, 0.99, "Vaccines do not cause autism. Original claim was retracted fraud."),
    (r"\b5g\b.{0,40}\b(spread|cause|transmit)\b.{0,30}\b(covid|virus|cancer)\b", False, 0.99, "5G cannot spread viruses or cause cancer."),
    (r"\bvaccine\w*\b.{0,40}\bmicrochip\w*\b", False, 0.99, "Vaccines do not contain microchips."),
    (r"\bdrinking bleach\b", False, 0.99, "Drinking bleach is deadly."),
    (r"\bclimate change\b.{0,30}\b(hoax|fake|fabricated|invented|scam)\b", False, 0.98, "Climate change denial contradicts scientific consensus."),
    (r"\bholoca?ust\b.{0,40}\b(hoax|fake|never happened|fabricated)\b", False, 0.99, "Holocaust denial is factually wrong."),
    (r"\bnarendra modi\b.{0,50}\b(is|was|serves as)\b.{0,20}\bprime minister\b", True, 0.99, "Narendra Modi is the Prime Minister of India."),
    (r"\bmodi\b.{0,20}\bprime minister of india\b", True, 0.99, "Narendra Modi is the Prime Minister of India."),
    (r"\bindia.{0,30}independence.{0,30}(1947|august 15)\b", True, 0.99, "India gained independence on August 15, 1947."),
    (r"\bearth\b.{0,30}\b(orbits|revolves around)\b.{0,20}\bsun\b", True, 0.99, "Earth orbiting the Sun is a basic astronomical fact."),
    (r"\bneil armstrong\b.{0,50}\b(moon|lunar|first)\b", True, 0.99, "Neil Armstrong was the first human on the Moon in 1969."),
    (r"\bworld war (ii|2|two)\b.{0,30}\b(ended|concluded)\b.{0,20}\b194[45]\b", True, 0.99, "World War II ended in 1945."),
    (r"\bwater\b.{0,20}\bh2o\b", True, 0.99, "Water's formula H2O is basic chemistry."),
    (r"\bdna\b.{0,30}\bgenetic\b", True, 0.99, "DNA carries genetic information."),
    (r"\b(deep state|illuminati|new world order)\b", False, 0.92, "Conspiracy theory language not grounded in evidence."),
    (r"\bchemtrail\w*\b", False, 0.95, "Chemtrail conspiracy is debunked."),
    (r"\bshare before (it.s )?deleted\b", False, 0.97, "Classic misinformation manipulation tactic."),
    (r"\bdoctors? (hate|don.t want|banned from telling)\b", False, 0.96, "Classic fake-health marketing pattern."),
    (r"\bcure.{0,20}(cancer|all disease).{0,30}(overnight|24 hours|instantly)\b", False, 0.99, "Instant disease cures are not medically possible."),
    (r"\bbig pharma\b.{0,30}\b(suppressing|hiding|covering up)\b", False, 0.90, "Pharma suppression conspiracy is unsubstantiated."),
    (r"\bmainstream media.{0,20}silent\b", False, 0.93, "Common misinformation framing device."),
]

def _check_knowledge_base(text: str) -> Optional[dict]:
    text_lower = text.lower()
    for pattern, is_real, confidence, reason in KNOWLEDGE_BASE:
        if _re.search(pattern, text_lower, _re.IGNORECASE):
            verdict = "REAL" if is_real else "FAKE"
            real_score = confidence if is_real else (1.0 - confidence)
            return {
                "verdict": verdict,
                "confidence": confidence,
                "real_score": real_score,
                "reasoning": reason,
                "red_flags": [] if is_real else [f"Knowledge base: {reason}"],
                "credibility_signals": [reason] if is_real else [],
                "claim_type": "factual",
                "recommendation": ("This aligns with established knowledge." if is_real else "This contradicts scientific/factual consensus. Do NOT share."),
                "_source": "knowledge_base",
            }
    return None

FAKE_SIGNALS = [
    ("breaking", 0.3), ("shocking", 0.3), ("exposed", 0.3), ("wake up", 0.35),
    ("share before deleted", 0.5), ("mainstream media silent", 0.45),
    ("big pharma", 0.4), ("deep state", 0.5), ("new world order", 0.55),
    ("illuminati", 0.55), ("whistleblower reveals", 0.3), ("they don't want you", 0.5),
    ("cover-up", 0.35), ("suppressed", 0.35), ("hidden truth", 0.4),
    ("miracle cure", 0.5), ("cures cancer", 0.45), ("doctors hate", 0.5),
    ("chemtrail", 0.55), ("microchip", 0.35), ("hoax", 0.4), ("conspiracy", 0.35),
    ("mind control", 0.5), ("5g tower", 0.45), ("government hiding", 0.45),
    ("elites", 0.3), ("globalist", 0.5), ("flat earth", 0.6),
    ("moon landing fake", 0.65), ("vaccines cause autism", 0.7), ("urgent alert", 0.4),
]

REAL_SIGNALS = [
    ("peer-reviewed", 0.4), ("clinical trial", 0.4), ("randomized controlled", 0.5),
    ("according to researchers", 0.35), ("published in nature", 0.5),
    ("published in lancet", 0.5), ("study finds", 0.3), ("scientists confirm", 0.3),
    ("researchers at", 0.25), ("official data", 0.3), ("government report", 0.25),
    ("statistics show", 0.3), ("journal of", 0.35), ("systematic review", 0.4),
    ("meta-analysis", 0.45), ("fda approved", 0.35), ("who reports", 0.3),
    ("confirmed by", 0.25), ("according to", 0.15), ("percent", 0.15),
]

def _heuristic_analysis(text: str) -> dict:
    text_lower = text.lower()
    kb = _check_knowledge_base(text)
    if kb:
        return kb
    fake_score = real_score = 0.0
    fake_hits = []
    real_hits = []
    for signal, weight in FAKE_SIGNALS:
        if signal in text_lower:
            fake_score += weight
            fake_hits.append(signal)
    for signal, weight in REAL_SIGNALS:
        if signal in text_lower:
            real_score += weight
            real_hits.append(signal)
    if fake_score == 0 and real_score == 0:
        real_score = 0.55
        fake_score = 0.45
    total = fake_score + real_score
    real_prob = real_score / total
    fake_prob = fake_score / total
    if real_prob > fake_prob:
        confidence = min(0.60 + real_score * 0.15, 0.94)
        verdict = "REAL"
    elif fake_prob > real_prob:
        confidence = min(0.60 + fake_score * 0.12, 0.97)
        verdict = "FAKE"
    else:
        confidence = 0.55
        verdict = "UNCERTAIN"
    real_score_out = confidence if verdict == "REAL" else (1.0 - confidence)
    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "real_score": round(real_score_out, 3),
        "reasoning": (
            f"Contains {len(fake_hits)} misinformation signals: {', '.join(fake_hits[:3])}." if verdict == "FAKE"
            else f"Contains {len(real_hits)} credibility markers: {', '.join(real_hits[:3])}." if verdict == "REAL"
            else "Mixed signals — manual verification recommended."
        ),
        "red_flags": [f"Signal: '{h}'" for h in fake_hits[:4]],
        "credibility_signals": [f"Marker: '{h}'" for h in real_hits[:3]],
        "claim_type": "conspiracy" if any(w in text_lower for w in ["illuminati", "deep state", "new world order"]) else "factual",
        "recommendation": "Do NOT share — likely misinformation." if verdict == "FAKE" else "Cross-check with reputable sources.",
        "_source": "heuristic",
    }

def llm_analyze(text: str) -> dict:
    if not text or not text.strip():
        return _heuristic_analysis("no text")
    kb = _check_knowledge_base(text)
    if kb:
        return kb
    gemini_result = _call_gemini(text)
    if gemini_result:
        return gemini_result
    return _heuristic_analysis(text)

def _ev_to_score(ev_result: dict) -> float:
    label = (ev_result or {}).get("label", "UNCERTAIN")
    return {"REAL": 0.82, "FAKE": 0.12, "UNCERTAIN": 0.50}.get(label, 0.50)

def _ml_to_real(ml_result: dict) -> float:
    if not ml_result:
        return 0.5
    return ml_result.get("real_prob", 50) / 100.0

def build_hybrid_verdict(ml_result, llm_result, ev_result):
    llm_source  = (llm_result or {}).get("_source", "heuristic")
    llm_real    = (llm_result or {}).get("real_score", 0.5)
    llm_verdict = (llm_result or {}).get("verdict", "UNCERTAIN")
    llm_conf    = (llm_result or {}).get("confidence", 0.7)
    ml_real  = max(0.05, min(0.95, _ml_to_real(ml_result)))
    ev_score = max(0.05, min(0.95, _ev_to_score(ev_result)))
    llm_real = max(0.05, min(0.95, llm_real))

    if llm_source == "knowledge_base":
        final_real = (llm_real * 0.70) + (ml_real * 0.20) + (ev_score * 0.10)
        if llm_verdict == "REAL":
            final_real = max(final_real, llm_conf * 0.95)
        elif llm_verdict == "FAKE":
            final_real = min(final_real, 1.0 - llm_conf * 0.95)
    elif llm_source == "gemini":
        final_real = (ml_real * 0.30) + (llm_real * 0.50) + (ev_score * 0.20)
    else:
        final_real = (ml_real * 0.55) + (llm_real * 0.25) + (ev_score * 0.20)

    final_fake = 1.0 - final_real
    if final_real >= 0.60:
        final_real = min(0.60 + (final_real - 0.60) * 1.6, 0.99)
        final_fake = 1.0 - final_real
    elif final_fake >= 0.60:
        final_fake = min(0.60 + (final_fake - 0.60) * 1.6, 0.99)
        final_real = 1.0 - final_fake

    if final_real >= 0.52:
        label = "REAL"
        conf  = final_real * 100
    elif final_fake >= 0.52:
        label = "FAKE"
        conf  = final_fake * 100
    else:
        label = (ml_result or {}).get("label", "UNCERTAIN") if ml_result else "UNCERTAIN"
        conf  = max(final_real, final_fake) * 100

    return {
        "label":          label,
        "confidence":     round(min(conf, 99.0), 1),
        "fake_prob":      round(final_fake * 100, 1),
        "real_prob":      round(final_real * 100, 1),
        "ml_score":       round(ml_real, 3),
        "llm_score":      round(llm_real, 3),
        "evidence_score": round(ev_score, 3),
        "final_score":    round(final_real, 3),
    }
