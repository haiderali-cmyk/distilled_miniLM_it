# Distilled MiniLM Italian Preprocessing Pipeline

This repository provides a preprocessing pipeline for building a **Italian text dataset** from Hugging Face datasets.  
It performs four sequential steps:

1. **Download** — fetch a dataset from Hugging Face (`gsarti/clean_mc4_it`) and save it to CSV.  
2. **Keyword Filter** — filter rows by Italian crime-related keywords.  
3. **Zero-Shot Classification** — further filter using a transformer-based zero-shot classifier (`facebook/bart-large-mnli`).  
4. **Sentence Chunking** — split long texts into sentences with token length constraints using spaCy + HuggingFace tokenizer.

The resulting CSV can then be used for downstream Italian NLP tasks such as classification, embeddings, or fine-tuning.


---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:haiderali-cmyk/distilled_miniLM_it.git
cd distilled_miniLM_it

# create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt
```
## Main Dependencies
``` bash
Python 3.9+
transformers
datasets
spaCy (it_core_news_lg)
PyYAML
pandas
tqdm
```
## Start Download Dataset
```bash
python pipeline.py --config config.yaml
```

### Output Files

clean_mc4_it_tiny.csv → raw downloaded dataset (tiny split).

tiny_filtered_data_from_csv.csv → keyword-filtered subset.

zero_shot_crime_only_output.csv → zero-shot classification filtered rows.

clean_mc4_it_sentence_chunks_latest.csv → final processed sentence-level dataset.

### Usage on Local Data

To use it on local dataset please add path of .csv file in .config.yaml file.
After that run python pipeline.pt --config _PATH_TO_CONFIG_FILE_

## Download Preprocessed Data
You can also download preprocessed data by clicking on below link:
https://drive.google.com/file/d/1qw9k9Rm9w20X0KEpc53MUp9oVGDJ8Vfx/view?usp=drive_link
