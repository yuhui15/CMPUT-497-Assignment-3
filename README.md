# CMPUT-497-Assignment-3
```
python -m spacy download es_core_news_lg
python -m spacy download fr_core_news_lg
python -m spacy download it_core_news_lg
python -m spacy download en_core_web_lg
python -m spacy download ro_core_news_lg
python -m spacy download zh_core_web_lg
python -m spacy download xx_ent_wiki_sm
pip install pandarallel
pip install sacremoses
pip install simalign
python ExpandNet/eval_release.py ExpandNet/res/data/se_gold_zh.tsv  expandnet_step3_project.out.tsv
python ExpandNet/eval_release.py ExpandNet/res/data/se_gold_zh.tsv LLM_zh.tsv

Step 1:
python3 expandnet_step1_translate.py \
  --src_data res/data/xlwsd_se13.xml \
  --lang_src en \
  --lang_tgt zh \
  --output_file expandnet_step1_translate.out.tsv

Step 2:
python3 expandnet_step2_align.py \
  --translation_df_file expandnet_step1_translate.out.tsv \
  --lang_src en \
  --lang_tgt zh \
  --aligner dbalign \
  --dict res/dicts/wikpan-en-zh.tsv \
  --output_file expandnet_step2_align.out.tsv

Step 3:
python3 expandnet_step3_project.py \
  --src_data res/data/xlwsd_se13.xml \
  --src_gold res/data/se13.key.txt \
  --dictionary res/dicts/wikpan-en-zh.tsv \
  --alignment_file expandnet_step2_align.out.tsv \
  --output_file expandnet_step3_project.out.tsv \
  --join_char _

```
