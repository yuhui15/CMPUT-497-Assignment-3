import re
import csv
from collections import defaultdict

# --------- I/O ---------
CEDICT_PATH = "cedict_ts.u8"
OUT_TSV     = "wikpan-en-zh.tsv"

# --------- CEDICT line regex ---------
CEDICT_LINE_RE = re.compile(r"""
    ^(?P<trad>[^ ]+)\s+
      (?P<simp>[^ ]+)\s+
      \[(?P<pinyin>[^\]]*)\]\s+
      /(?P<defs>.+)/\s*$
""", re.X)

# --------- cleaning helpers ---------
PARENS_BRACKETS_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")  # remove (...) or [...]
WHITES_RE = re.compile(r"\s+")

def strip_parens_and_brackets(text: str) -> str:
    """Remove all (...) and [...] substrings (with their contents)."""
    return PARENS_BRACKETS_RE.sub("", text)

def split_glosses(defs_raw: str):
    """
    CC-CEDICT definitions are slash-separated.
    After stripping, further split by semicolons ';' into finer glosses.
    """
    items = []
    for part in defs_raw.split("/"):
        part = part.strip()
        if not part:
            continue
        # remove bracketed notes then split by semicolon
        part = strip_parens_and_brackets(part)
        for sub in part.split(";"):
            g = sub.strip()
            if g:
                items.append(g)
    return items

def truncate_to_token(rule_text: str) -> str:
    """
    Scan left-to-right, keep only chars in [A-Za-z0-9-_].
    On the first char NOT in that set, remove that char and everything to the right.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    out = []
    for ch in rule_text:
        if ch in allowed:
            out.append(ch)
        else:
            break
    return "".join(out).lower()

def remove_extra_punct(text: str) -> str:
    """
    Remove ALL punctuation except letters, digits, underscore, and space.
    (This runs after truncate_to_token.)
    """
    return re.sub(r"[^a-zA-Z0-9_\s]", "", text)

def gloss_to_single_token(gloss: str) -> str:
    """
    Normalize a gloss into ONE token per the rule.
    - collapse inner whitespace first (so we can easily detect multi-word)
    - then truncate by allowed charset rule
    - remove all other punctuation (except _ and space)
    - ensure the result has no spaces and is non-empty
    """
    gloss = WHITES_RE.sub(" ", gloss.strip())
    token = truncate_to_token(gloss)
    token = remove_extra_punct(token)
    if not token or " " in token:
        return ""
    return token

# --------- main pipeline ---------
def main():
    en2zh = defaultdict(set)

    with open(CEDICT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            m = CEDICT_LINE_RE.match(line.strip())
            if not m:
                continue
            zh = m.group("simp").strip()
            defs_raw = m.group("defs")

            # 1) split into gloss pieces
            pieces = split_glosses(defs_raw)
            # 2) to single token by the truncation rule + punctuation cleaning
            for g in pieces:
                tok = gloss_to_single_token(g)
                if tok:
                    en2zh[tok].add(zh)

    # write TSV
    rows = [(en, " ".join(sorted(zset))) for en, zset in en2zh.items() if en]
    rows.sort(key=lambda x: x[0])

    with open(OUT_TSV, "w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w, delimiter="\t")
        for en, zh_space in rows:
            wr.writerow([en, zh_space])

    print(f"[OK] wrote {OUT_TSV}  rows={len(rows)}")

if __name__ == "__main__":
    main()
