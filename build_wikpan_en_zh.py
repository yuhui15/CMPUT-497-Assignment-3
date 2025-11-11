import os
import re
import csv
import json
import time
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import requests

# Avoid transformers pulling in torchvision (not used here, but safe guard)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# ---------------------------
# Config (can be overridden)
# ---------------------------
DEFAULT_MODEL = "open-mistral-7b"
CACHE_JSON = "wikpan_en_zh_cache.json"

SYSTEM_PROMPT = (
    "You are a bilingual lexicon expert. "
    "Given a Chinese lemma and a list of English glosses describing its meanings, "
    "output EXACTLY ONE English lemma as a SINGLE lowercase token that best represents "
    "the core sense across these glosses. If glosses are multi-word, choose a common "
    "single-word hypernym or the most fitting single-word head. "
    "Return only the single English word. No punctuation or explanations."
)

USER_PROMPT_TMPL = (
    "Chinese lemma: {zh}\n"
    "English glosses: {glosses}\n\n"
    "Output: (ONE lowercase English word)"
)


# ---------------------------
# Mistral API client
# ---------------------------
class MistralClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = (api_key or os.getenv("MISTRAL_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing Mistral API key. Set env var MISTRAL_API_KEY.")
        self.model = model
        self.base = "https://api.mistral.ai/v1"
        self.sess = requests.Session()
        self.sess.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def chat(self, system: str, user: str, temperature: float = 0.0,
             top_p: float = 1.0, max_tokens: int = 8, retries: int = 4) -> str:
        url = f"{self.base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        backoff = 1.0
        for attempt in range(retries + 1):
            resp = self.sess.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            # transient?
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            # show server body
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise requests.HTTPError(f"Mistral error {resp.status_code}: {body}", response=resp)


# ---------------------------
# CEDICT parsing
# ---------------------------
CEDICT_LINE_RE = re.compile(r"""
    ^(?P<trad>[^ ]+)\s+
     (?P<simp>[^ ]+)\s+
     \[(?P<pinyin>[^\]]*)\]\s+
     /(?P<defs>.+)/\s*$
""", re.X)

def parse_cedict(path: str) -> Dict[str, List[str]]:
    """
    Parse cedict_ts.u8 into dict: zh_simplified -> list of English gloss strings.
    Each slash-separated definition is one gloss entry.
    """
    z2g: Dict[str, List[str]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            m = CEDICT_LINE_RE.match(line.strip())
            if not m:
                continue
            simp = m.group("simp")
            defs_raw = m.group("defs")  # "to plan/to design/scheme"
            glosses = [d.strip() for d in defs_raw.split("/") if d.strip()]
            if glosses:
                z2g[simp].extend(glosses)
    return z2g


# ---------------------------
# Normalization helpers
# ---------------------------
def normalize_en_single_token(text: str) -> str:
    """
    Force to a SINGLE lowercase English token:
      - take only letters a-z
      - pick the first word
      - fallback to empty on failure
    """
    if not text:
        return ""
    # lowercase + strip non-letters to spaces, then split
    t = text.lower()
    t = re.sub(r"[^a-z]+", " ", t)
    t = t.strip()
    if not t:
        return ""
    return t.split()[0]

def normalize_zh_token(text: str) -> str:
    """
    Chinese side: keep as-is, but replace internal spaces with underscores (wikpan convention).
    """
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", "_", t)
    return t


# ---------------------------
# Main pipeline
# ---------------------------
def build_wikpan_en_zh(cedict_path: str,
                       out_tsv: str,
                       model: str = DEFAULT_MODEL,
                       sleep: float = 0.8,
                       max_items: Optional[int] = None,
                       min_zh_len: int = 1,
                       cache_path: str = CACHE_JSON):
    """
    1) Parse CEDICT: zh -> [glosses...]
    2) For each zh, ask Mistral to compress gloss list into ONE english token
    3) Aggregate: en_token -> {zh1, zh2, ...}
    4) Write wikpan-en-zh.tsv
    """
    print(f"[INFO] Loading CEDICT from: {cedict_path}")
    zh2gloss = parse_cedict(cedict_path)
    print(f"[INFO] CEDICT entries (unique zh): {len(zh2gloss):,}")

    # load cache
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    client = MistralClient(model=model)

    en2zh: Dict[str, set] = defaultdict(set)

    count = 0
    for zh, glosses in zh2gloss.items():
        if len(zh) < min_zh_len:
            continue
        if max_items is not None and count >= max_items:
            break

        # build user prompt with *deduped* concise gloss list (limit to avoid overlong prompts)
        uniq_gloss = list(dict.fromkeys(glosses))[:12]  # trim long lists
        prompt = USER_PROMPT_TMPL.format(zh=zh, glosses="; ".join(uniq_gloss))

        if zh in cache and cache[zh]:
            raw = cache[zh]
        else:
            raw = client.chat(SYSTEM_PROMPT, prompt, temperature=0.0, top_p=1.0, max_tokens=8)
            cache[zh] = raw
            time.sleep(sleep)

        en_token = normalize_en_single_token(raw)
        zh_tok = normalize_zh_token(zh)

        if en_token and zh_tok:
            en2zh[en_token].add(zh_tok)
            count += 1

        # progress log
        if count % 200 == 0:
            print(f"[{count}] {zh_tok} -> {en_token}")

    # write TSV
    rows = []
    for en, zset in en2zh.items():
        zlist = sorted(zset)
        rows.append((en, " ".join(zlist)))
    rows.sort(key=lambda x: x[0])

    with open(out_tsv, "w", encoding="utf-8", newline="") as w:
        tsvw = csv.writer(w, delimiter="\t")
        for en, zh_space_join in rows:
            tsvw.writerow([en, zh_space_join])

    # save cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"[OK] Wrote wikpan file: {out_tsv}  (rows={len(rows):,})")


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cedict", required=True, help="Path to cedict_ts.u8")
    ap.add_argument("--out", default="wikpan-en-zh.tsv", help="Output TSV file")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Mistral model name (default open-mistral-7b)")
    ap.add_argument("--sleep", type=float, default=0.8, help="Sleep seconds between API calls")
    ap.add_argument("--max_items", type=int, default=None, help="Limit number of zh items to process (for testing)")
    ap.add_argument("--min_zh_len", type=int, default=1, help="Filter very short zh tokens")
    ap.add_argument("--cache", default=CACHE_JSON, help="Cache JSON path")
    args = ap.parse_args()

    build_wikpan_en_zh(
        cedict_path=args.cedict,
        out_tsv=args.out,
        model=args.model,
        sleep=args.sleep,
        max_items=args.max_items,
        min_zh_len=args.min_zh_len,
        cache_path=args.cache
    )

if __name__ == "__main__":
    main()
