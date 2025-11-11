import os
import re
import csv
import json
import time
from typing import Dict, List, Optional

# Hugging Face Transformers (local inference)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========= Input / Output paths ==========
CEDICT_PATH = "cedict_ts.u8"            # raw CC-CEDICT
OUT_TSV     = "wikpan-en-zh.tsv"        # output dictionary
CACHE_JSON  = "wikpan_cache.json"       # caching

# ========= Local model config ==========
# You can use a local dir (e.g., "C:/models/Mistral-7B-Instruct") after you `git lfs clone` it,
# or a hub ID (requires allowlist + license): "mistralai/Mistral-7B-Instruct-v0.2"
LOCAL_MODEL_ID = os.getenv("LOCAL_MISTRAL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

# Generation defaults: keep short to bias the model to output a single token/word
MAX_NEW_TOKENS = 3
TEMPERATURE    = 0.0
TOP_P          = 1.0
REPETITION_PEN = 1.05  # light penalty helps avoid echoing prompt text

# Throughput pacing (avoid thrashing on CPU; tune for your hardware)
SLEEP_EACH_SEC = 0.05

# ========= Prompt config (same spirit as your baseline) ==========
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

# ========= CC-CEDICT parsing ==========
CEDICT_LINE_RE = re.compile(r"""
    ^(?P<trad>[^ ]+)\s+
      (?P<simp>[^ ]+)\s+
      \[(?P<pinyin>[^\]]*)\]\s+
      /(?P<defs>.+)/\s*$
""", re.X)

def clean_token(raw: str) -> str:
    """
    Keep characters until the first illegal symbol.
    Illegal symbol = any char NOT in [A-Za-z0-9-_].

    If the first char is illegal -> return empty string.
    """
    if not raw:
        return ""

    raw = raw.strip().lower()

    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    result = []

    for ch in raw:
        if ch in allowed:
            result.append(ch)
        else:
            break  # stop at first invalid symbol

    return "".join(result)


def parse_cedict(path: str) -> Dict[str, List[str]]:
    """
    Parse CC-CEDICT into dict: zh_simplified -> list of English glosses.
    We keep ALL glosses per Han entry (duplicates removed later).
    """
    zh2gloss: Dict[str, List[str]] = {}
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
                zh2gloss.setdefault(simp, []).extend(glosses)
    return zh2gloss

# ========= Normalization ==========
def normalize_en_word(text: str) -> str:
    return clean_token(text)

def normalize_zh(text: str) -> str:
    """
    Replace inner spaces with underscores (wikpan convention).
    CEDICT lemmas usually have no spaces, but keep this for safety.
    """
    if not text:
        return ""
    return re.sub(r"\s+", "_", text.strip())

# ========= Local Mistral wrapper ==========
class LocalMistral:
    """
    Minimal local inference wrapper for Mistral-7B-Instruct via Transformers.
    - Loads tokenizer/model with device_map='auto' and bfloat16 if supported.
    - Uses the chat template with system+user roles to format the prompt.
    - Returns raw text (post-processing is done by caller).
    """
    def __init__(self, model_id: str = LOCAL_MODEL_ID):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True
        )

        # Choose dtype and device map automatically
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto"  # let Accelerate place layers on available GPUs/CPU
        )

    def generate_once(
        self,
        system_msg: str,
        user_msg: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        repetition_penalty: float = REPETITION_PEN
    ) -> str:
        """
        Format messages with the model's chat template, then run .generate().
        Return the decoded new text after the prompt.
        """
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]

        # Apply chat template. For Mistral instruct, this inserts special tokens/format.
        # return_tensors='pt' gives you a tensor ready for generation.
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # We only need the newly generated part
        gen_ids = out[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()

# ========= Main build function ==========
def build_dictionary():
    print(f"[INFO] Loading CEDICT: {CEDICT_PATH}")
    zh2gloss = parse_cedict(CEDICT_PATH)
    print(f"[INFO] Chinese entries: {len(zh2gloss)}")
    print(f"[INFO] Loading local model: {LOCAL_MODEL_ID}")

    client = LocalMistral(LOCAL_MODEL_ID)

    # Load cache to avoid repeated generations
    cache: Dict[str, str] = {}
    if os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass

    # Accumulator: english_head -> set of chinese_lemmas
    en2zh: Dict[str, set] = {}

    count = 0
    for zh, glosses in zh2gloss.items():
        # Deduplicate and trim gloss list to keep prompts short and focused
        uniq_gloss = list(dict.fromkeys(glosses))[:12]
        gloss_str = "; ".join(uniq_gloss)

        if zh in cache:
            raw = cache[zh]
        else:
            user_prompt = USER_PROMPT_TEMPLATE.format(zh=zh, glosses=gloss_str)
            raw = client.generate_once(
                system_msg=SYSTEM_PROMPT,
                user_msg=user_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            cache[zh] = raw

        en_word = normalize_en_word(raw)
        zh_word = normalize_zh(zh)
        print(en_word,zh_word)

        if not en_word:
            continue

        en2zh.setdefault(en_word, set()).add(zh_word)
        count += 1

        # Progress log
        if count % 200 == 0:
            print(f"[{count}] {zh_word} -> {en_word}")

    # Write TSV (sorted by English headword)
    rows = [(en, " ".join(sorted(zset))) for en, zset in en2zh.items()]
    rows.sort(key=lambda x: x[0])

    with open(OUT_TSV, "w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w, delimiter="\t")
        for en_word, zh_space_join in rows:
            wr.writerow([en_word, zh_space_join])


    print(f"[OK] Dictionary generated → {OUT_TSV} (rows={len(rows)})")

# ========= Run ==========
if __name__ == "__main__":
    build_dictionary()
