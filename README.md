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
git clone https://github.com/haiderali-cmyk/distilled_miniLM_it.git
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
datasets==2.21.0
spaCy
PyYAML
pandas
tqdm
```
## Start Download Dataset
```bash
python data_preprocessing.py --config config.yaml
```

### Output Files

data.csv → raw downloaded dataset (tiny split).

data_filtered.csv → keyword-filtered subset.

data_filtered_zero_shot.csv → zero-shot classification filtered rows.

data_filtered_zero_shot_chunk.csv → final processed chunk-level dataset.

## Configurations

### Preprocessing Steps
Developers can easily skip any step by just removing that step from steps in config.yaml file. Except download, it's important for loading data.
```bash
  steps:
    - download # Do not remove this it's important for loading data
    - keyword_filter
    - zero_shot
    - process
```

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
  local_csv: PATH_TO_THE_CSV
```

### Using mc4-it data

Please note: if developer want to download the dataset they should set the value of local_csv to false. In that case mc4 italian data will be downloaded.
```bash
local_csv: false
```

So, if local_csv value is set to path script will use local_csv from the specified path. In case local_csv value is set to false, it will download mc4-it data from hugging face.

### Keywords Filtering
Developers can update or add new keywords in .config.yml file. For that please refer to keyword_filter step in .config.yml file. By default chunk_size is set to 100,000 which means script will process 100,000 chunks at a time.
```bash
keyword_filter:
  chunk_size: 100000
  keywords:
    - "omicidio"
    - "assassinio"
    - "femminicidio"
    - "aggressione"
    - "lesioni personali"
    - "violenza sessuale"
    - "stupro"
    - "omicidio colposo"
    - "tentato omicidio"
    - "furto"
    - "rapina"
    - "scasso"
    - "borseggio"
    - "effrazione"
    - "spaccio di droga"
    - "traffico di stupefacenti"
    - "detenzione di droga"
    - "riciclaggio di denaro"
    - "corruzione"
    - "estorsione"
    - "truffa"
    - "usura"
    - "frode fiscale"
    - "evasione fiscale"
    - "appropriazione indebita"
    - "pedofilia"
    - "abuso su minori"
    - "pornografia minorile"
    - "crimine organizzato"
    - "associazione mafiosa"
    - "mafia"
    - "camorra"
    - "ndrangheta"
    - "terrorismo"
    - "attentato terroristico"
    - "radicalizzazione"
    - "cellula terroristica"
    - "sequestro di persona"
    - "minacce"
    - "intimidazione"
    - "latitante"
    - "indagine penale"
    - "procedimento penale"
    - "arresto"
    - "carcere"
    - "custodia cautelare"
```

### Zero-shot classification
For zero-shot classification configuration please refer to step 3 (i.e. zero_shot). By default cuda:device is set to 0. Developer can easily configure the settings
in config.yml file. 
```bash
zero_shot:
  model_name: "facebook/bart-large-mnli"
  device: 0 
  chunk_size: 1000
  batch_size: 8
  labels: ["crimine", "non crimine"]
  hypothesis_template: "Questo testo riguarda {}."
```

### Sentence Chunking
For sentence chunking please refer to process in config.yml file. We use 'multilingual-e5-large' model for tokenization because of it's best performance. We set
minimum and maximum token limit to make sure we don't use too large or too small chunks for distillation process.

```bash
process:
  spacy_model: "it_core_news_lg"
  tokenizer_name: "intfloat/multilingual-e5-large"
  chunksize: 100
  max_tokens: 250
  min_tokens: 80
  output_column: "sentence1"
  auto_download_spacy_model: true
```

### Redo steps or Continue
In case data preprocessing is stopped in any step, developer can easily continue from the last step. For that, developer need to ensure that previous step files have been saved properly. In case, developers want to redo every step again, they can easily set force to true.
```bash
general:
  force: false  # If true, redo steps even if outputs exist
  # Ordered list of steps to run when CLI --steps is not provided
```



## Download Preprocessed Data
You can also download preprocessed data by clicking on below link:
https://drive.google.com/file/d/1qw9k9Rm9w20X0KEpc53MUp9oVGDJ8Vfx/view?usp=drive_link
