"""
LLM Baseline (Chinese) using Mistral 7B, with BN evaluation output.

Inputs
------
1) se13.key.txt
   - Source-side gold with BabelNet synset IDs at token level.
   - Format per line:
       d000.s000.t000 bn:00041942n
     (instance_id<TAB>bn_id)

2) semeval2013.key.WNversion.txt
   - Same instances but labeled with WordNet sensekeys.
   - Format per line:
       d000.s000.t000 group%1:03:00::            # one sensekey is typical
     (instance_id<TAB>wn_sensekey[ wn_sensekey ...])

What this script does
---------------------
• Align the two files by instance_id to build a mapping:
      WordNet synset (e.g., 'group.n.01') → most frequent BN synset (e.g., 'bn:00041942n')
• For each *unique* WordNet synset in the mapping:
      - Fetch its English gloss from WordNet
      - Prompt Mistral 7B (LLM baseline) to produce ONE Chinese lemma
• Write a BN-evaluable TSV:
      bn_synset_id<TAB>Chinese_lemma
  (One line per WN synset that could be mapped to a BN synset.)

Notes
-----
• The LLM sees ONLY the English gloss (no gold labels, no target lemmas).
• Multi-word outputs are normalized with underscores, spaces trimmed.
• Rate-limited to 1 req/s by default (change SLEEP_EACH_SEC if needed).
"""

import os
import re
import time
import json
import csv
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import requests

# ============== User-configurable paths ==============
PATH_BN   = "se13.key.txt"                     # e.g., d000.s000.t000 <TAB> bn:00041942n
PATH_WN   = "semeval2013.key.WNversion.txt"   # e.g., d000.s000.t000 <TAB> group%1:03:00::
OUT_TSV   = "LLM_zh_BN.tsv"                   # final BN-evaluable baseline file
CACHE_JSON = "llm_cache.json"                 # optional: cache {wn_synset_name: chinese_lemma}

# ============== Mistral API config (reusing your client pattern) ==============
API_KEY = "1EhmID66Dsd7LlWZSbGFp38SV6Hsg1BZ"   # <-- put your key here (you provided it)
MISTRAL_MODEL = "open-mistral-7b"
SLEEP_EACH_SEC = 1.0                            # 1 req/sec as a safe default

# ============== LLM Prompt config ==============
SYSTEM_PROMPT = (
    "You are a bilingual lexicon expert. Given an English WordNet gloss "
    "that defines a specific sense (synset), output exactly ONE Chinese lemma "
    "that best matches this definition. Output ONLY the Chinese word or short phrase. "
    "Do not output punctuation, quotes, or explanations."
)

USER_PROMPT_TEMPLATE = (
    "English gloss:\n"
    "{gloss}\n\n"
    "Output: (ONE Chinese lemma only)"
)

# ============== WordNet (NLTK) ==============
# We only rely on NLTK to convert WN sensekeys to synsets and read glosses.
from nltk.corpus import wordnet as wn
import nltk

def ensure_nltk():
    """Ensure WordNet is available."""
    try:
        _ = wn.synsets("bank")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


# ============== Minimal Mistral client (Chat Completions) ==============
class APIMistral:
    """Minimal client for Mistral Chat Completions API (compatible with your previous style)."""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not self.api_key:
            self.api_key = API_KEY
        self.base_url = "https://api.mistral.ai/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0, top_p: float = 0.9, max_tokens: int = 16) -> str:
        """Send prompts to Mistral and return *raw* content string."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        resp = self.session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ============== Parsing helpers ==============
def read_bn_key(path: str) -> Dict[str, str]:
    """
    Read BN key file: instance_id -> bn_id.
    If multiple BN IDs per line exist, take the first.
    """
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            inst = parts[0]
            # pick first BN id that looks like bn:XXXX
            bn = next((p for p in parts[1:] if p.startswith("bn:")), None)
            if bn:
                mapping[inst] = bn
    return mapping


def read_wn_version(path: str) -> Dict[str, str]:
    """
    Read WN version file: instance_id -> first WordNet sensekey on the line.
    (If there are multiple sensekeys, we take the first. In SemEval WNversion
     files this typically corresponds to that instance's target token.)
    """
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            inst = parts[0]
            # take the first token that looks like a WN sensekey: has a '%' inside
            key = next((p for p in parts[1:] if "%" in p), None)
            if key:
                mapping[inst] = key
    return mapping


def wnkey_to_synset_name(wn_sensekey: str) -> Optional[str]:
    """
    Convert a WordNet sensekey (e.g., 'group%1:03:00::') to a synset name ('group.n.01').
    Return None if conversion fails.
    """
    try:
        lemma = wn.lemma_from_key(wn_sensekey)
        return lemma.synset().name()  # e.g., 'group.n.01'
    except Exception:
        return None


def normalize_zh_lemma(text: str) -> str:
    """
    Post-process LLM outputs to a clean Chinese lemma:
    - strip spaces and quotes
    - replace inner spaces with underscores (for MWEs)
    - drop trailing punctuation
    - keep only the first line / token-like segment
    """
    if not text:
        return ""
    # keep only first line
    text = text.strip().splitlines()[0].strip()
    # remove surrounding quotes and punctuation-like wrappers
    text = text.strip(" \t\"'`[](){}，。；；、：:；,.")
    # collapse whitespace to single space then replace space with underscore
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", "_")
    # keep it reasonably short
    return text[:64]


# ============== Build WN synset → BN synset mapping via instance alignment ==============
def build_wn_to_bn_map(bn_by_inst: Dict[str, str],
                       wnkey_by_inst: Dict[str, str]) -> Dict[str, str]:
    """
    Join by instance_id first, then:
      wn_synset_name -> Counter of bn_ids
    Return the most-frequent BN id per WN synset.
    """
    bucket: Dict[str, Counter] = defaultdict(Counter)

    common = set(bn_by_inst.keys()) & set(wnkey_by_inst.keys())
    for inst in common:
        bn_id = bn_by_inst[inst]
        wn_key = wnkey_by_inst[inst]
        wn_syn = wnkey_to_synset_name(wn_key)
        if wn_syn and bn_id:
            bucket[wn_syn][bn_id] += 1

    # choose the most frequent BN per WN synset
    wn2bn: Dict[str, str] = {}
    for wn_syn, counter in bucket.items():
        bn_id, _ = counter.most_common(1)[0]
        wn2bn[wn_syn] = bn_id
    return wn2bn


# ============== Main ==============
def main():
    ensure_nltk()

    # 1) Load keys
    bn_by_inst = read_bn_key(PATH_BN)
    wnkey_by_inst = read_wn_version(PATH_WN)

    if not bn_by_inst:
        raise RuntimeError(f"No BN instances read from: {PATH_BN}")
    if not wnkey_by_inst:
        raise RuntimeError(f"No WN sensekeys read from: {PATH_WN}")

    # 2) Build WN synset → BN synset mapping
    wn2bn = build_wn_to_bn_map(bn_by_inst, wnkey_by_inst)
    if not wn2bn:
        raise RuntimeError("WN→BN mapping is empty; check that your two key files have overlapping instance_ids.")

    # 3) Prepare cache (optional)
    cache: Dict[str, str] = {}
    if os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # 4) Mistral client
    api = APIMistral(model_name=MISTRAL_MODEL, api_key=API_KEY)

    # 5) Iterate over unique WN synsets (keys of wn2bn), query LLM, write BN-evaluable TSV
    done, total = 0, len(wn2bn)
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as w:
        tsvw = csv.writer(w, delimiter="\t")
        # no header expected by evaluator: just bn_id<TAB>zh_lemma
        for wn_syn, bn_syn in wn2bn.items():
            # fetch gloss for this WN synset
            try:
                gloss = wn.synset(wn_syn).definition()
            except Exception:
                # as a fallback, try to reconstruct via any sensekey example (rarely needed)
                continue

            # use cache if available
            if wn_syn in cache and cache[wn_syn]:
                zh_lemma = cache[wn_syn]
            else:
                # Build user prompt and call the LLM
                user_prompt = USER_PROMPT_TEMPLATE.format(gloss=gloss)
                raw = api.generate(SYSTEM_PROMPT, user_prompt, temperature=0.0, top_p=0.9, max_tokens=16)
                zh_lemma = normalize_zh_lemma(raw)
                # store in cache
                cache[wn_syn] = zh_lemma
                # be a good citizen
                time.sleep(SLEEP_EACH_SEC)

            if not zh_lemma:
                # skip empty predictions
                continue

            # Write BN-evaluable line: bn_id<TAB>Chinese_lemma
            tsvw.writerow([bn_syn, zh_lemma])

            done += 1
            if done % 50 == 0 or done == total:
                print(f"[{done}/{total}] {wn_syn} -> {bn_syn} :: {zh_lemma}")

    # 6) Save cache
    try:
        with open(CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"✅ Done. Wrote BN-evaluable baseline to: {OUT_TSV}")


if __name__ == "__main__":
    main()
