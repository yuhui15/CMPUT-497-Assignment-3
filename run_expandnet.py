import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

def run(cmd, cwd=None):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"[ERROR] Command failed: {' '.join(cmd)}")
    if res.stdout.strip():
        print(res.stdout)
    if res.stderr.strip():
        print(res.stderr, file=sys.stderr)
    return res

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser(description="ExpandNet pipeline runner")
    p.add_argument("--repo_dir", required=True,
                   help="Directory containing expandnet_step*.py and eval_release.py")
    # Step 1 (Translate)
    p.add_argument("--src_xml", help="Source XML file for Step 1 translate (if omitted, Step 1 is skipped)")
    p.add_argument("--lang_src", default="en", help="Source language key (default: en)")
    p.add_argument("--lang_tgt", required=True, help="Target language key, e.g., es, fr, zh")
    p.add_argument("--translation_tsv", help="Provide translation TSV to skip Step 1")
    # Step 2 (Align)
    p.add_argument("--aligner", default="dbalign", choices=["dbalign", "simalign"],
                   help="Aligner to use (default: dbalign)")
    p.add_argument("--dict", required=True,
                   help="Path to bilingual dict TSV (or 'bn' if code supports BabelNet)")
    # Step 3 (Project)
    p.add_argument("--src_gold", required=True,
                   help="Source-language gold key file (e.g., res/data/se13.key.txt)")
    p.add_argument("--join_char", default="_", help="Join character for MWEs (default: _)")
    # Eval (optional)
    p.add_argument("--eval_gold", help="Gold file for target language (synset\\tlemmas...) to evaluate against")
    # Work/output
    p.add_argument("--work_dir", default="work", help="Working directory for outputs (default: work)")
    p.add_argument("--python", default=sys.executable, help="Python executable to call (default: current)")
    return p.parse_args()

def main():
    args = parse_args()
    repo = Path(args.repo_dir).resolve()
    work = Path(args.work_dir).resolve()
    ensure_dir(work)

    # Resolve script paths
    step1 = repo / "expandnet_step1_translate.py"
    step2 = repo / "expandnet_step2_align.py"
    step3 = repo / "expandnet_step3_project.py"
    eval_script = repo / "eval_release.py"

    for s in [step1, step2, step3]:
        if not s.exists():
            raise SystemExit(f"[ERROR] Missing script: {s}")

    # Step 1 → translation TSV
    if args.translation_tsv:
        translation_tsv = Path(args.translation_tsv).resolve()
        if not translation_tsv.exists():
            raise SystemExit(f"[ERROR] Provided translation TSV not found: {translation_tsv}")
    else:
        if not args.src_xml:
            raise SystemExit("[ERROR] Either --translation_tsv OR --src_xml must be provided.")
        translation_tsv = work / "expandnet_step1_translate.out.tsv"
        cmd = [
            args.python, str(step1),
            "--src_data", args.src_xml,
            "--lang_src", args.lang_src,
            "--lang_tgt", args.lang_tgt,
            "--output_file", str(translation_tsv),
        ]
        run(cmd)

    # Step 2 → alignment TSV
    align_tsv = work / "expandnet_step2_align.out.tsv"
    cmd = [
        args.python, str(step2),
        "--translation_df_file", str(translation_tsv),
        "--lang_src", args.lang_src,
        "--lang_tgt", args.lang_tgt,
        "--aligner", args.aligner,
        "--dict", args.dict,
        "--output_file", str(align_tsv),
    ]
    run(cmd)

    # Step 3 → projection TSV
    project_out = work / "expandnet_step3_project.out.tsv"
    cmd = [
        args.python, str(step3),
        "--src_data", args.src_xml if args.src_xml else "",   # most repos expect the original XML used in step1
        "--src_gold", args.src_gold,
        "--dictionary", args.dict,
        "--alignment_file", str(align_tsv),
        "--output_file", str(project_out),
        "--join_char", args.join_char,
    ]
    # Some repos require --src_data; if user skipped step1, they must still pass the original XML they used.
    # If empty value causes issues in your repo, raise here:
    if not args.src_xml:
        print("[WARN] --src_xml was not provided; if expandnet_step3_project.py requires it, pass it explicitly.")
    run(cmd)

    print(f"\n[OK] Projection output: {project_out}")

    # Optional: eval
    if args.eval_gold:
        if not eval_script.exists():
            print(f"[WARN] No eval script found at {eval_script}. Skipping evaluation.")
        else:
            print("\n[Eval] Running eval_release.py...")
            cmd = [args.python, str(eval_script), args.eval_gold, str(project_out)]
            run(cmd)

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
