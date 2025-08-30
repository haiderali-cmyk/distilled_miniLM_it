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
transformers
datasets
spaCy
PyYAML
pandas
tqdm
```
## Start Download Dataset
```bash
python pipeline.py --config config.yaml
```

### Output Files

data.csv → raw downloaded dataset (tiny split).

data_filtered.csv → keyword-filtered subset.

data_filtered_zero_shot.csv → zero-shot classification filtered rows.

data_filtered_zero_shot_chunk.csv → final processed chunk-level dataset.

### Usage on Local Data

To use it on local dataset please add path of .csv file in .config.yaml file.

## Configuration
### Using local data
Developers can easily use their own data for all processing steps. For this, they should provide
path of .csv file.
```bash
download:
  hf_dataset: "gsarti/clean_mc4_it"
  config_name: "tiny"
  split: "train"
  streaming: false
  # Optional: provide an absolute path to a pre-downloaded CSV to skip downloading step
  # The CSV must contain a column named as in `io.text_column` (default: "text")
  # local_csv: /home/niche-3/Documents/haiderali/best_model/Preprocessing/data/sample_300.csv
  local_csv: PATH_TO_THE_CSV
```
Please note: if developer want to download the dataset they should set the value of local_csv to false. In that case mc4 italian data will be downloaded.
```bash
local_csv: false
```
So, if local_csv value is set to path script will use local_csv from the specified path. In case local_csv value is set to false, it will download mc4-it data from hugging face.

### Preprocessing Steps
Developers can easily skip any step by just removing that step from steps in config.yaml file
```bash
  steps:
    - download
    - keyword_filter
    - zero_shot
    - process
```
Developers can either download the dataset from Hugging Face or use a locally downloaded copy. To use a downloaded dataset, simply add the dataset path to the local_csv key.

```bash
download:
  hf_dataset: "gsarti/clean_mc4_it"
  config_name: "tiny"
  split: "train"
  streaming: false
  # Optional: provide an absolute path to a pre-downloaded CSV to skip Hugging Face download
  # The CSV must contain a column named as in `io.text_column` (default: "text")
  local_csv: None # Specify path to the .csv file. If not path is specified than data will be downloaded from hugging face.
```

## Download Preprocessed Data
You can also download preprocessed data by clicking on below link:
https://drive.google.com/file/d/1qw9k9Rm9w20X0KEpc53MUp9oVGDJ8Vfx/view?usp=drive_link
