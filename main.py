from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import httpx
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

# ══════════════════════════════════════════════════════════════════════════
# ENGLISH + LEET NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════

LEET_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "@": "a", "$": "s", "!": "i"
}

SCAM_WORDS_EN = {
    "urgent": 2, "verify": 2, "click": 1, "account": 1, "suspended": 3,
    "bank": 2, "password": 3, "login": 2, "immediately": 2, "winner": 3,
    "prize": 3, "otp": 4, "congratulations": 2, "free": 1, "limited": 1,
    "expire": 2, "blocked": 2, "unusual": 2, "confirm": 1, "update": 1,
}

SCAM_PHRASES_EN = [
    (r"click.*link", 3), (r"verify.*account", 4), (r"suspended.*immediately", 5),
    (r"won.*prize", 4), (r"send.*otp", 5), (r"confirm.*password", 4),
    (r"account.*blocked", 4), (r"unusual.*activity", 3), (r"update.*details", 3),
    (r"limited.*time", 2),
]

# ══════════════════════════════════════════════════════════════════════════
# ARABIC NORMALIZATION + DETECTION
# ══════════════════════════════════════════════════════════════════════════

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: strip diacritics, unify alef/yeh/teh marbuta variants."""
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)   # unify alef forms
    text = re.sub(r"ى", "ي", text)         # alef maksura -> yeh
    text = re.sub(r"ة", "ه", text)         # teh marbuta -> heh
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    return text

# Arabic scam keywords (GCC-relevant), weighted same way as English
SCAM_WORDS_AR = {
    "عاجل": 2,                  # urgent
    "تاكيد": 2,                 # confirm/verify
    "تحقق": 2,                  # verify
    "حساب": 1,                  # account
    "تعليق": 3,                 # suspension
    "موقوف": 3,                 # suspended/blocked
    "بنك": 2,                   # bank
    "كلمه المرور": 3,            # password
    "دخول": 2,                  # login
    "فورا": 2,                  # immediately
    "فوري": 2,                  # immediately
    "فزت": 3,                   # you won
    "جائزه": 3,                 # prize
    "رمز التحقق": 4,             # OTP/verification code
    "مبروك": 2,                 # congratulations
    "مجاني": 1,                 # free
    "محدود": 1,                 # limited
    "ينتهي": 2,                 # expires
    "محظور": 2,                 # blocked
  "نشاط غير معتاد": 3,         # unusual activity
    "تحديث بياناتك": 3,          # update your data
    "اضغط هنا": 3,               # click here
    "ارسل": 1,                  # send
    "بيانات بطاقتك": 4,          # your card details
}

SCAM_PHRASES_AR = [
    (r"اضغط.*رابط", 3),          # click link
    (r"تحقق.*حساب", 4),          # verify account
    (r"حساب.*موقوف", 4),         # account suspended
    (r"فزت.*جائزه", 4),          # won a prize
    (r"ارسل.*رمز", 5),           # send code (OTP)
    (r"تاكيد.*كلمه المرور", 4),   # confirm password
    (r"حساب.*محظور", 4),         # account blocked
    (r"نشاط.*غير معتاد", 3),     # unusual activity
    (r"حدث.*بياناتك", 3),        # update your data
    (r"عرض.*محدود", 2),          # limited offer
]


def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


# ══════════════════════════════════════════════════════════════════════════
# URL / LINK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

URL_PATTERN = re.compile(
    r"((?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)"
)
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "shorturl.at", "cutt.ly",
}
SUSPICIOUS_TLDS = re.compile(r"\.(xyz|top|click|loan|work|gq|tk|ml|ga|cf|cam|surf)$")

# Brands commonly impersonated in GCC scams + their real domains
LOOKALIKE_TARGETS = {
    "paypal": "paypal.com",
    "omantel": "omantel.om",
    "bankmuscat": "bankmuscat.com",
    "ooredoo": "ooredoo.om",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "google": "google.com",
    "whatsapp": "whatsapp.com",
    "instagram": "instagram.com",
}


def extract_domain(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = "http://" + url
    match = re.match(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1).lower() if match else url.lower()


def looks_like_lookalike(domain: str) -> str | None:
    """Detect typosquatting / lookalike domains for known brands."""
    base = domain.split(".")[0]
    for brand in LOOKALIKE_TARGETS:
        if brand == base:
            continue
        # crude similarity: shares most characters, isn't the real domain
        if brand in base and domain != LOOKALIKE_TARGETS[brand]:
            return brand
        # char-substitution check (e.g. paypaI, paypa1, paypall)
        if len(base) >= len(brand) - 1 and sum(1 for a, b in zip(base, brand) if a == b) >= len(brand) - 2:
            if domain != LOOKALIKE_TARGETS[brand] and brand[:4] in base:
                return brand
    return None


async def analyze_url(url: str) -> dict:
    """Fetch a URL with strict timeout/redirect limits and inspect the result."""
    domain = extract_domain(url)
    findings = {"domain": domain, "score": 0, "signals": []}

    if domain in SHORTENER_DOMAINS:
        findings["score"] += 2
        findings["signals"].append("link shortener")

    if SUSPICIOUS_TLDS.search(domain):
        findings["score"] += 3
        findings["signals"].append("suspicious domain extension")

    lookalike = looks_like_lookalike(domain)
    if lookalike:
        findings["score"] += 5
        findings["signals"].append(f"looks like fake '{lookalike}' domain")

    # Real fetch — strict timeout, limited redirects, no content download beyond headers
    try:
        async with httpx.AsyncClient(follow_redirects=True, max_redirects=4, timeout=4.0) as client:
            resp = await client.head(f"http://{domain}", timeout=4.0)
            final_domain = extract_domain(str(resp.url))
            if final_domain != domain:
                findings["signals"].append(f"redirects to {final_domain}")
                if looks_like_lookalike(final_domain):
                    findings["score"] += 3
            findings["reachable"] = True
            findings["status"] = resp.status_code
    except Exception:
        findings["reachable"] = False
        findings["signals"].append("could not verify destination (unreachable or blocked)")

    return findings


# ══════════════════════════════════════════════════════════════════════════
# CORE SCAN LOGIC
# ══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(LEET_MAP.get(ch, ch) for ch in text)
    text = normalize_arabic(text)
    # keep arabic + latin + digits + space
    return re.sub(r"[^\u0600-\u06FFa-z0-9 ]", " ", text)


@app.post("/scan")
async def scan(message: Message):
    raw = message.text
    normalized = normalize_text(raw)
    has_arabic = contains_arabic(raw)

    score = 0
    signals = []

    # English keyword + phrase scoring
    for word, weight in SCAM_WORDS_EN.items():
        if re.search(r"\b" + word + r"\b", normalized):
            score += weight
            signals.append(word)

    for pattern, weight in SCAM_PHRASES_EN:
        if re.search(pattern, normalized):
            score += weight
            label = pattern.replace(r"\b", "").replace(".*", " + ")
            if label not in signals:
                signals.append(label)

    # Arabic keyword + phrase scoring
    if has_arabic:
        for word, weight in SCAM_WORDS_AR.items():
            if word in normalized:
                score += weight
                signals.append(word)
        for pattern, weight in SCAM_PHRASES_AR:
            if re.search(pattern, normalized):
                score += weight
                label = pattern.replace(".*", " + ")
                if label not in signals:
                    signals.append(label)

    # Link extraction + analysis (pattern-level always; live fetch with timeout budget)
    urls = URL_PATTERN.findall(raw)
    urls = [u[0] if isinstance(u, tuple) else u for u in urls]
    urls = list(dict.fromkeys(urls))[:3]  # cap at 3 URLs to keep response fast

    link_details = []
    if urls:
        score += 1
        signals.append("contains link")
        results = await asyncio.gather(*(analyze_url(u) for u in urls), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                continue
            score += r["score"]
            signals.extend(r["signals"])
            link_details.append(r)

    signals = list(dict.fromkeys(signals))

    # Raw score is uncapped internally; convert to a 0-100 risk scale.
    # Cap raw score at 25 (realistic ceiling given current weights) then scale to 100.
    raw_capped = min(score, 25)
    score_100 = round((raw_capped / 25) * 100)

    if score_100 >= 40:
        result = "Scam"
        reason = "High-risk patterns detected — do not interact"
    elif score_100 >= 20:
        result = "Suspicious"
        reason = "Some scam signals found — proceed with caution"
    else:
        result = "Safe"
        reason = "No strong scam indicators found"

    return {
        "result": result,
        "score": score_100,
        "signals": signals,
        "reason": reason,
        "max_score": 100,
        "language_detected": "ar+en" if has_arabic else "en",
        "links_analyzed": link_details,
    }
