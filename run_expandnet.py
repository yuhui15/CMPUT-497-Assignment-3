import os
import sys
import subprocess
from pathlib import Path

# =======================
# CONFIG: EDIT AS NEEDED
# =======================

CONFIG = {
    # Path to the ExpandNet repo (contains expandnet_step*.py & eval_release.py)
    "REPO_DIR": "ExpandNet",

    # Data files (defaults mirror the examples you provided)
    "SRC_XML": "xlwsd_se13.xml",       # Step1 input XML
    "SRC_GOLD": "se13.key.txt",        # Step3 source gold (english BabelNet keys)
    "DICT_TSV": "ExpandNet/res/dicts/wikpan-en-es.tsv",   # DBAlign dictionary (change to your target)
    "EVAL_GOLD": None,                          # Optional: target-language gold, e.g., res/data/se_gold_es.tsv

    # Language codes
    "LANG_SRC": "en",
    "LANG_TGT": "es",                           # change to your target language (e.g., "zh", "fr", "ro")

    # Aligner: "dbalign" or "simalign"
    "ALIGNER": "dbalign",

    # Work/output dir
    "WORK_DIR": "",

    # Python executable (default: current)
    "PY_EXE": sys.executable,

    # If you already have a translation TSV, set this path to skip Step1; else keep as None
    "TRANSLATION_TSV": None,  # e.g., "work_es/expandnet_step1_translate.out.tsv"
}

# ============== helpers ==============
def run(cmd, cwd=None):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout, end="")
    if res.stderr:
        print(res.stderr, file=sys.stderr, end="")
    if res.returncode != 0:
        raise SystemExit(f"[ERROR] Command failed: {' '.join(cmd)}")
    return res

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    REPO = Path(CONFIG["REPO_DIR"]).resolve()
    WORK = Path(CONFIG["WORK_DIR"]).resolve()
    ensure_dir(WORK)

    # Resolve scripts
    step1 = REPO / "expandnet_step1_translate.py"
    step2 = REPO / "expandnet_step2_align.py"
    step3 = REPO / "expandnet_step3_project.py"
    eval_py = REPO / "eval_release.py"

    for s in [step1, step2, step3]:
        if not s.exists():
            raise SystemExit(f"[ERROR] Missing script: {s}")

    # ========== Step 1: Translate (optional if TRANSLATION_TSV provided) ==========
    if CONFIG["TRANSLATION_TSV"]:
        translation_tsv = Path(CONFIG["TRANSLATION_TSV"]).resolve()
        if not translation_tsv.exists():
            raise SystemExit(f"[ERROR] TRANSLATION_TSV not found: {translation_tsv}")
        print(f"[INFO] Skip Step1. Use existing translation: {translation_tsv}")
    else:
        translation_tsv = WORK / "expandnet_step1_translate.out.tsv"
        cmd = [
            CONFIG["PY_EXE"], str(step1),
            "--src_data", CONFIG["SRC_XML"],
            "--lang_src", CONFIG["LANG_SRC"],
            "--lang_tgt", CONFIG["LANG_TGT"],
            "--output_file", str(translation_tsv),
        ]
        run(cmd, cwd=str(REPO))

    # ========== Step 2: Align ==========
    align_tsv = WORK / "expandnet_step2_align.out.tsv"
    cmd = [
        CONFIG["PY_EXE"], str(step2),
        "--translation_df_file", str(translation_tsv),
        "--lang_src", CONFIG["LANG_SRC"],
        "--lang_tgt", CONFIG["LANG_TGT"],
        "--aligner", CONFIG["ALIGNER"],
        "--dict", CONFIG["DICT_TSV"],
        "--output_file", str(align_tsv),
    ]
    run(cmd, cwd=str(REPO))

    # ========== Step 3: Project ==========
    project_out = WORK / "expandnet_step3_project.out.tsv"
    cmd = [
        CONFIG["PY_EXE"], str(step3),
        "--src_data", CONFIG["SRC_XML"],
        "--src_gold", CONFIG["SRC_GOLD"],
        "--dictionary", CONFIG["DICT_TSV"],
        "--alignment_file", str(align_tsv),
        "--output_file", str(project_out),
        "--join_char", "_",
    ]
    run(cmd, cwd=str(REPO))
    print(f"[OK] Projection output: {project_out}")

    # ========== Optional: Eval ==========
    if CONFIG["EVAL_GOLD"]:
        if not eval_py.exists():
            print(f"[WARN] eval_release.py not found at {eval_py}; skip evaluation.")
        else:
            print("[Eval] Running eval_release.py ...")
            cmd = [CONFIG["PY_EXE"], str(eval_py), CONFIG["EVAL_GOLD"], str(project_out)]
            run(cmd, cwd=str(REPO))

    print("✅ Done.")

if __name__ == "__main__":
    main()
