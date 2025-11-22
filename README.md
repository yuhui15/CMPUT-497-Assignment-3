# CMPUT-497-Assignment-3
```
The command to build the environment:
pip install pandarallel
pip install sacremoses
pip install simalign
pip install spacy
pip install sentencepiece
python3 -m spacy download es_core_news_lg
python3 -m spacy download fr_core_news_lg
python3 -m spacy download it_core_news_lg
python3 -m spacy download en_core_web_lg
python3 -m spacy download ro_core_news_lg
python3 -m spacy download zh_core_web_lg
python3 -m spacy download xx_ent_wiki_sm

The command to run:

cd Expandnet

python3 expandnet_step1_translate.py --src_data res/data/xlwsd_se13.xml --lang_src en --lang_tgt zh --mymemory_email your.email@example.com --sleep_sec 1.0

python3 expandnet_step2_align.py --translation_df_file expandnet_step1_translate.out.tsv --lang_src en --lang_tgt zh --aligner dbalign --dict res/dicts/wikpan-en-zh.tsv --output_file expandnet_step2_align.out.tsv

python3 expandnet_step3_project.py --src_data res/data/xlwsd_se13.xml --src_gold res/data/se13.key.txt --dictionary res/dicts/wikpan-en-zh.tsv --alignment_file expandnet_step2_align.out.tsv --output_file expandnet_step3_project.out.tsv --join_char _

cd ..

python3 ExpandNet/eval_release.py ExpandNet/res/data/se_gold_zh.tsv  ExpandNet/expandnet_step3_project.out.tsv

python3 ExpandNet/eval_release.py ExpandNet/res/data/se_gold_zh.tsv LLM_zh.tsv
```
