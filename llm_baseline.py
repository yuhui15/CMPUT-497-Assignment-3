import os
import re
import time
import json
import csv
from typing import Dict, List, Optional, Tuple

import requests
from nltk.corpus import wordnet as wn
import nltk

# ============== User-configurable paths ==============
PATH_BN   = "se13.key.txt"                     # d000.s000.t000 <TAB> bn:00041942n
PATH_WN   = "semeval2013.key.WNversion.txt"   # d000.s000.t000 <TAB> group%1:03:00::
OUT_TSV   = "LLM_zh.tsv"                   # BN-evaluable baseline file (line-by-line)
CACHE_JSON = "llm_cache.json"                 # {wn_synset_name: chinese_lemma}
TARGET_LANGUAGE = "simplified Chinese"

# ============== Mistral API config ==============
API_KEY = os.getenv("MISTRAL_API_KEY", "AIVUFuS9Js7QkBJ3RufabHlrKgeNUR4a")  # set env var or replace
MISTRAL_MODEL = "open-mistral-7b"             # try "mistral-small-latest" if needed
SLEEP_EACH_SEC = 0.1                           # 1 req/sec

# ============== LLM Prompt config ==============
INSTRUCTION = (
"You are a bilingual lexicon expert."
"Given a dictionary definition, produce the single word in {TARGET_LANGUAGE} that best matches this definition."
"Provide only the {TARGET_LANGUAGE} word without explanations!"
)

USER_PROMPT_TEMPLATE = (
    "English gloss:\n{gloss}\n\n"
    "Output: (ONE Chinese lemma only)"
)

def build_full_prompt(gloss: str) -> str:
    """Concatenate instruction + task content into a single user message."""
    return f"{INSTRUCTION}\n\n{USER_PROMPT_TEMPLATE.format(gloss=gloss)}"

# ============== NLTK ensure ==============
def ensure_nltk():
    """Ensure WordNet corpora are available."""
    try:
        _ = wn.synsets("bank")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

# ============== Minimal Mistral client (Chat Completions) ==============
class APIMistral:
    """
    Minimal client for Mistral Chat Completions API with:
    - single 'user' message (schema-safe),
    - Accept header,
    - helpful error body,
    - exponential backoff on transient errors.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = str(model_name)
        self.api_key = (api_key or os.getenv("MISTRAL_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing Mistral API key. Set MISTRAL_API_KEY env var or pass api_key.")
        self.base_url = "https://api.mistral.ai/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 32,
        retries: int = 4
    ) -> str:
        """Send prompt as a single user message. Retries on 429/5xx with exponential backoff."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        url = f"{self.base_url}/chat/completions"

        backoff = 1.0
        for attempt in range(retries + 1):
            resp = self.session.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

            # Surface server explanation
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text

            # Retry on transient errors
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff)
                backoff *= 2
                continue

            # Otherwise, raise with full body to debug (e.g., 400 payload issue)
            raise requests.HTTPError(
                f"Mistral API error {resp.status_code}. Body={detail}",
                response=resp
            )

# ============== IO helpers (preserve line order) ==============
def read_bn_key_lines(path: str) -> List[Tuple[str, str]]:
    """
    Read BN key lines preserving order.
    Returns a list of (instance_id, bn_id).
    If multiple BN IDs present, take the first that starts with 'bn:'.
    """
    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            inst = parts[0]
            bn = next((p for p in parts[1:] if p.startswith("bn:")), None)
            if bn:
                rows.append((inst, bn))
    return rows

def read_wn_key_dict(path: str) -> Dict[str, str]:
    """
    Read WNversion into a dict: instance_id -> first sensekey on the line.
    """
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            inst = parts[0]
            key = next((p for p in parts[1:] if "%" in p), None)
            if key:
                mapping[inst] = key
    return mapping

# ============== text normalization ==============
def normalize_zh_lemma(text: str) -> str:
    """
    Post-process LLM outputs to a clean Chinese lemma:
    - keep only the first line
    - strip quotes / punctuation
    - spaces -> underscores (for MWEs)
    """
    if not text:
        return ""
    text = text.strip().splitlines()[0].strip()
    text = text.strip(" \t\"'`[](){}，。；、：:,.")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", "_")
    if "_" in text:
        text = text.split("_", 1)[0]
    return text[:64]

# ============== Main (line-by-line pairing) ==============
def main():
    ensure_nltk()

    # Load BN lines (ordered) and WN dict (for lookup)
    bn_rows = read_bn_key_lines(PATH_BN)         # List[(inst, bn)]
    wn_dict = read_wn_key_dict(PATH_WN)          # inst -> sensekey

    if not bn_rows:
        raise RuntimeError(f"No BN instances read from: {PATH_BN}")
    if not wn_dict:
        raise RuntimeError(f"No WN sensekeys read from: {PATH_WN}")

    # Optional cache to avoid repeated LLM calls for same WN synset
    cache: Dict[str, str] = {}
    if os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    api = APIMistral(model_name=MISTRAL_MODEL, api_key=API_KEY)

    done = 0
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as w:
        tsvw = csv.writer(w, delimiter="\t")

        # Iterate line-by-line over BN file order
        for inst, bn_id in bn_rows:
            wn_key = wn_dict.get(inst)
            if not wn_key:
                # No matching WN line for this instance_id
                continue

            # Convert sensekey -> synset name
            try:
                lemma = wn.lemma_from_key(wn_key)
                wn_syn = lemma.synset().name()  # e.g., 'group.n.01'
                gloss = lemma.synset().definition()
                print(gloss)
            except Exception:
                continue

            # Check cache per WN synset (optional but efficient)
            if wn_syn in cache and cache[wn_syn]:
                zh_lemma = cache[wn_syn]
            else:
                prompt = build_full_prompt(gloss)
                raw = api.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=10)
                zh_lemma = normalize_zh_lemma(raw)
                cache[wn_syn] = zh_lemma
                time.sleep(SLEEP_EACH_SEC)

            if not zh_lemma:
                continue

            # Write exactly one line per (BN instance line) → Chinese lemma
            tsvw.writerow([bn_id, zh_lemma])

            # Echo the exact line to terminal
            print(f"{bn_id}\t{zh_lemma}", flush=True)

            done += 1

    print(f"✅ Done. Wrote BN-evaluable baseline (line-by-line) to: {OUT_TSV}")

if __name__ == "__main__":
    main()
