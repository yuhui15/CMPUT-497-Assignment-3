import os
import re
import csv
import json
import time
from typing import Dict, List, Optional
import requests

# ========= Same API style as baseline ==========
API_KEY = os.getenv("MISTRAL_API_KEY", "AIVUFuS9Js7QkBJ3RufabHlrKgeNUR4a")  # replace or export
MISTRAL_MODEL = "open-mistral-7b"
SLEEP_EACH_SEC = 0.1

# ========= Input / Output paths ==========
CEDICT_PATH = "cedict_ts.u8"            # raw CC-CEDICT
OUT_TSV = "wikpan-en-zh.tsv"            # output dictionary
CACHE_JSON = "wikpan_cache.json"        # caching

# ========= Prompt config (same style as baseline) ==========
SYSTEM_PROMPT = (
    "You are a bilingual lexicon expert. "
    "Given a Chinese lemma and several English glosses, "
    "produce ONE English lemma as a single lowercase word that best captures the core sense. "
    "Return only ONE lowercase English word. No punctuation, no explanation."
)

USER_PROMPT_TEMPLATE = (
    "Chinese lemma: {zh}\n"
    "English glosses: {glosses}\n\n"
    "Output: (ONE lowercase English lemma)"
)

# ========= Mistral client (identical style to your baseline) ==========
class APIMistral:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = str(model_name)
        self.api_key = (api_key or os.getenv("MISTRAL_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing Mistral API key.")
        self.base_url = "https://api.mistral.ai/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def generate(self, system_msg: str, user_msg: str,
                 temperature: float = 0.0, top_p: float = 1.0,
                 max_tokens: int = 1, retries: int = 4) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        url = f"{self.base_url}/chat/completions"

        backoff = 1.0
        for attempt in range(retries + 1):
            resp = self.session.post(url, json=payload, timeout=60)

            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()

            # retry for transient errors
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff)
                backoff *= 2
                continue

            # fatal error: surface whole body
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise requests.HTTPError(
                f"Mistral API error {resp.status_code}. Body={detail}",
                response=resp
            )

# ========= Parse CC-CEDICT ==========
CEDICT_LINE_RE = re.compile(r"""
    ^(?P<trad>[^ ]+)\s+
      (?P<simp>[^ ]+)\s+
      \[(?P<pinyin>[^\]]*)\]\s+
      /(?P<defs>.+)/\s*$
""", re.X)

def parse_cedict(path: str) -> Dict[str, List[str]]:
    """
    Parse CC-CEDICT into dict: zh_simplified -> list of English glosses.
    """
    zh2gloss = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            m = CEDICT_LINE_RE.match(line.strip())
            if not m:
                continue
            simp = m.group("simp")
            defs_raw = m.group("defs")
            glosses = [d.strip() for d in defs_raw.split("/") if d.strip()]
            if glosses:
                if simp not in zh2gloss:
                    zh2gloss[simp] = []
                zh2gloss[simp].extend(glosses)
    return zh2gloss

# ========= Normalization ==========
def normalize_en_word(text: str) -> str:
    """
    Convert LLM output into one valid lowercase English lemma.
    Exactly identical logic style as your baseline normalization.
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z]+", " ", t).strip()
    if not t:
        return ""
    return t.split()[0]

def normalize_zh(text: str) -> str:
    """
    Replace inner spaces with underscores for wikpan convention.
    """
    if not text:
        return ""
    return re.sub(r"\s+", "_", text.strip())

# ========= Main build function ==========
def build_dictionary():
    print(f"[INFO] Loading CEDICT: {CEDICT_PATH}")
    zh2gloss = parse_cedict(CEDICT_PATH)
    print(f"[INFO] Chinese entries: {len(zh2gloss)}")

    # Load cache
    cache = {}
    if os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass

    api = APIMistral(model_name=MISTRAL_MODEL, api_key=API_KEY)

    en2zh = {}

    count = 0
    for zh, glosses in zh2gloss.items():
        uniq_gloss = list(dict.fromkeys(glosses))
        gloss_str = "; ".join(uniq_gloss[:10])  # limit prompt size

        if zh in cache:
            raw = cache[zh]
        else:
            prompt = USER_PROMPT_TEMPLATE.format(zh=zh, glosses=gloss_str)
            raw = api.generate(SYSTEM_PROMPT, prompt)
            print(raw)
            cache[zh] = raw
            time.sleep(SLEEP_EACH_SEC)

        en_word = normalize_en_word(raw)
        zh_word = normalize_zh(zh)

        if not en_word:
            continue

        if en_word not in en2zh:
            en2zh[en_word] = set()
        en2zh[en_word].add(zh_word)

        count += 1
        if count % 200 == 0:
            print(f"[{count}] {zh_word} -> {en_word}")

    # write TSV
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w, delimiter="\t")
        for en_word in sorted(en2zh.keys()):
            wr.writerow([en_word, " ".join(sorted(en2zh[en_word]))])

    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"[OK] Dictionary generated → {OUT_TSV}")

# ========= Run ==========
if __name__ == "__main__":
    build_dictionary()
