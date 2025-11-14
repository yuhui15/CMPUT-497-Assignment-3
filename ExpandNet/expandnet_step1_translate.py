import argparse
import time
import pandas as pd
import spacy
from deep_translator import MyMemoryTranslator
import xml_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
    parser.add_argument(
        "--src_data",
        type=str,
        default="xlwsd_se13.xml",
        help="Path to the XLWSD XML corpus file.",
    )
    parser.add_argument(
        "--lang_src",
        type=str,
        default="en",
        help="Source language (short code, e.g. en, zh, fr, es).",
    )
    parser.add_argument(
        "--lang_tgt",
        type=str,
        default="fr",
        help="Target language (short code, e.g. en, zh, fr, es).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="expandnet_step1_translate.out.tsv",
        help="File to store sentences and translations.",
    )
    # Email for MyMemory API (required to get the larger quota)
    parser.add_argument(
        "--mymemory_email",
        type=str,
        required=True,
        help="Email address for the MyMemory API (required).",
    )
    # Sleep between translation requests
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=1.0,
        help="Seconds to sleep between translation requests (default: 1.0).",
    )
    return parser.parse_args()


# Map short language codes (used in CLI) to MyMemory language names/codes
MYMEMORY_LANG_MAP = {
    # English
    "en": "english",
    "en-gb": "english",
    "en-us": "english",
    "en-ca": "english",
    "en-au": "english",

    # Chinese
    "zh": "chinese simplified",
    "zh-cn": "chinese simplified",
    "zh-hans": "chinese simplified",
    "zh-tw": "chinese traditional",
    "zh-hant": "chinese traditional",

    # Some common extras
    "fr": "french",
    "es": "spanish",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
}


# spaCy model mapping – **these must actually exist in your environment**
SPACY_MODEL_MAP = {
    "en": "en_core_web_lg",
    # FIX: use a real Chinese model name
    "zh": "zh_core_web_sm",
    "fr": "fr_core_news_lg",
    "es": "es_core_news_lg",
}


def load_spacy_pipeline(lang_code: str):
    """
    Load the spaCy pipeline for a given short language code (e.g. 'en', 'zh').

    Raises a clear RuntimeError if the model is not available instead of
    silently failing and returning raw strings later.
    """
    model_name = SPACY_MODEL_MAP.get(lang_code)
    if model_name is None:
        raise RuntimeError(
            f"No spaCy model mapping defined for language '{lang_code}'. "
            f"Please extend SPACY_MODEL_MAP."
        )
    try:
        print(f"Loading spaCy model for '{lang_code}': {model_name}")
        return spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"Cannot load spaCy model '{model_name}' for language '{lang_code}'.\n"
            f"Make sure it is installed, e.g.:\n"
            f"  python -m spacy download {model_name}"
        ) from e


def main():
    # Parse the arguments.
    args = parse_args()

    print(f"Languages (CLI): {args.lang_src} -> {args.lang_tgt}")
    print(f"Corpus:          {args.src_data}")
    print(f"Output file:     {args.output_file}")
    print(f"MyMemory email:  {args.mymemory_email}")
    print(f"Sleep between requests: {args.sleep_sec} seconds")

    # Load the data.
    df_src = xml_utils.process_xml(args.src_data)
    print(f"Data loaded: {len(df_src)} rows")

    df_sent = xml_utils.extract_sentences(df_src)
    print(f"Sentences assembled: {len(df_sent)} rows")

    # --- Translator setup (MyMemory) ---
    # Keep CLI short codes, but map them for MyMemoryTranslator ONLY
    src_key = args.lang_src.lower()
    tgt_key = args.lang_tgt.lower()

    mm_src = MYMEMORY_LANG_MAP.get(src_key, src_key)
    mm_tgt = MYMEMORY_LANG_MAP.get(tgt_key, tgt_key)

    print(f"MyMemory languages (mapped): {mm_src} -> {mm_tgt}")

    try:
        translator = MyMemoryTranslator(
            source=mm_src,
            target=mm_tgt,
            email=args.mymemory_email,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize MyMemoryTranslator for {args.lang_src} -> {args.lang_tgt} "
            f"(mapped as {mm_src} -> {mm_tgt}): {e}"
        )

    # --- spaCy pipelines (for token/lemma on BOTH sides) ---
    pipelines = {}
    pipelines[args.lang_src] = load_spacy_pipeline(args.lang_src)
    pipelines[args.lang_tgt] = load_spacy_pipeline(args.lang_tgt)

    # Chinese doesn't use lemmatization in your pipeline
    lemmatize_tgt = False if args.lang_tgt in ["zh"] else True

    def tokenize_sentence(sentence: str, lang: str, do_lemmatize: bool = False):
        """
        Tokenize (and optionally lemmatize) a sentence using the spaCy pipeline
        for a given language.
        """
        nlp = pipelines[lang]
        doc = nlp(sentence)
        if do_lemmatize:
            return " ".join(token.lemma_.replace(" ", "_") for token in doc)
        else:
            return " ".join(token.text.replace(" ", "_") for token in doc)

    # --- Translate using MyMemory with throttling ---
    texts = df_sent["text"].tolist()
    translations = []

    print("Starting translation with MyMemory...")

    for i, sent in enumerate(texts, start=1):
        try:
            tr = translator.translate(sent)
        except Exception as e:
            print(f"[WARN] Translation failed for sentence index {i-1}: {e}")
            tr = ""  # or use original sentence: tr = sent

        translations.append(tr)

        # Progress logging
        if i % 50 == 0 or i == len(texts):
            print(f"  Translated {i}/{len(texts)} sentences")

        # Sleep between requests to avoid hitting the API too hard
        time.sleep(args.sleep_sec)

    df_sent["translation"] = translations

    # --- Tokenization / lemmatization on the *target* side ---
    df_sent["translation_token"] = df_sent["translation"].apply(
        lambda s: tokenize_sentence(s, args.lang_tgt, False)
    )

    df_sent["translation_lemma"] = df_sent["translation"].apply(
        lambda s: tokenize_sentence(s, args.lang_tgt, lemmatize_tgt)
    )

    print(f"Translation complete: {len(df_sent)} sentences processed\n")

    print(f'Saving to "{args.output_file}"...')
    cols = [
        "sentence_id",
        "text",
        "translation",
        "lemma",
        "translation_token",
        "translation_lemma",
    ]
    df_sent[cols].to_csv(args.output_file, sep="\t", index=False)
    print("Complete!")


if __name__ == "__main__":
    main()
