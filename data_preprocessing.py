import os
import csv
import math
import logging
from pathlib import Path
from typing import Dict, List, Any

import yaml
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from transformers import pipeline as hf_pipeline, AutoTokenizer

import spacy
from spacy.cli import download as spacy_download


# ------------------------------
# Config management
# ------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
	with open(config_path, "r", encoding="utf-8") as f:
		config = yaml.safe_load(f)
	return config


def build_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
	base_dir = Path(cfg["io"]["base_dir"]).expanduser().resolve()
	base_dir.mkdir(parents=True, exist_ok=True)
	paths = {
		"download_csv": str(base_dir / cfg["io"]["download_csv"]),
		"filtered_csv": str(base_dir / cfg["io"]["filtered_csv"]),
		"zero_shot_csv": str(base_dir / cfg["io"]["zero_shot_csv"]),
		"processed_csv": str(base_dir / cfg["io"]["processed_csv"]),
	}
	return paths


# ------------------------------
# Step 1: Download tiny dataset
# ------------------------------

def step_download(cfg: Dict[str, Any], out_csv: str) -> None:
	if os.path.exists(out_csv) and not cfg.get("general", {}).get("force", False):
		logging.info(f"Download output exists at {out_csv}; skipping.")
		return

	logging.info("Downloading the dataset (this may take a while)...")
	ds_cfg = cfg["download"]
	dataset = load_dataset(
		ds_cfg["hf_dataset"],
		ds_cfg.get("config_name", None),
		split=ds_cfg.get("split", "train"),
		streaming=bool(ds_cfg.get("streaming", False))
	)[:300]

	total_rows = len(dataset)
	logging.info(f"Loaded {total_rows} examples from '{ds_cfg.get('config_name', 'default')}' split.")

	fieldnames = dataset.column_names
	logging.info(f"Writing to CSV: {out_csv}")
	with open(out_csv, mode="w", newline='', encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=fieldnames,
			quoting=csv.QUOTE_MINIMAL,
			escapechar='\\'
		)
		writer.writeheader()
		for row in tqdm(dataset, total=total_rows, desc="Writing rows"):
			writer.writerow({key: str(row.get(key, "")).replace('\r', ' ').replace('\n', ' ') for key in fieldnames})

	logging.info("Download step completed.")


# ------------------------------
# Step 2: Keyword filter
# ------------------------------

def step_keyword_filter(cfg: Dict[str, Any], in_csv: str, out_csv: str) -> None:
	if os.path.exists(out_csv) and not cfg.get("general", {}).get("force", False):
		logging.info(f"Keyword-filter output exists at {out_csv}; skipping.")
		return

	io_text_col = cfg["io"]["text_column"]
	kw_cfg = cfg["keyword_filter"]
	chunk_size = int(kw_cfg.get("chunk_size", 100000))
	keywords = [str(k).lower() for k in kw_cfg.get("keywords", [])]

	logging.info(f"Keyword-filtering {in_csv} -> {out_csv} with {len(keywords)} keywords")
	matched_rows = 0
	total_rows = 0

	with pd.read_csv(in_csv, chunksize=chunk_size) as reader:
		for i, chunk in enumerate(reader):
			logging.info(f"Processing chunk {i+1} of keyword filtering")
			chunk[io_text_col] = chunk[io_text_col].astype(str).str.lower()
			mask = chunk[io_text_col].apply(lambda x: any(keyword in x for keyword in keywords))
			filtered_chunk = chunk[mask]

			matched_rows += len(filtered_chunk)
			total_rows += len(chunk)

			mode = 'w' if i == 0 else 'a'
			header = (i == 0)
			filtered_chunk.to_csv(out_csv, index=False, mode=mode, header=header)

	logging.info(f"Keyword filter finished. {matched_rows} / {total_rows} rows matched.")


# ------------------------------
# Step 3: Zero-shot classification
# ------------------------------

def step_zero_shot(cfg: Dict[str, Any], in_csv: str, out_csv: str) -> None:
	if os.path.exists(out_csv) and not cfg.get("general", {}).get("force", False):
		logging.info(f"Zero-shot output exists at {out_csv}; skipping.")
		return

	zs_cfg = cfg["zero_shot"]
	io_text_col = cfg["io"]["text_column"]
	chunk_size = int(zs_cfg.get("chunk_size", 1000))
	batch_size = int(zs_cfg.get("batch_size", 8))
	labels = zs_cfg.get("labels", ["crimine", "non crimine"])  # order matters
	hypothesis_template = zs_cfg.get("hypothesis_template", "Questo testo riguarda {}.")
	device_index = int(zs_cfg.get("device", 3))

	logging.info("Loading zero-shot model on GPU device=%s", device_index)
	classifier = hf_pipeline("zero-shot-classification", model=zs_cfg.get("model_name", "facebook/bart-large-mnli"), device=device_index)

	with pd.read_csv(in_csv, chunksize=chunk_size) as reader:
		for i, chunk in enumerate(reader):
			logging.info(f"Classifying chunk {i+1} ({len(chunk)} rows)")
			chunk = chunk.dropna(subset=[io_text_col]).reset_index(drop=True)
			texts: List[str] = chunk[io_text_col].astype(str).tolist()
			results: List[dict] = []

			for j in tqdm(range(0, len(texts), batch_size), desc=f"Zero-shot batch {i+1}"):
				batch = texts[j:j+batch_size]
				try:
					batch_results = classifier(batch, candidate_labels=labels, hypothesis_template=hypothesis_template)
					if isinstance(batch_results, dict):
						batch_results = [batch_results]
					for k, result in enumerate(batch_results):
						top_label = result["labels"][0]
						if top_label == labels[0]:  # treat first label as positive class
							row_idx = j + k
							if row_idx < len(chunk):
								row = chunk.iloc[row_idx]
								results.append(row.to_dict())
				except Exception as e:
					logging.warning(f"Error in batch {j}-{j+batch_size}: {e}")
					continue

			result_df = pd.DataFrame(results)
			if not result_df.empty:
				mode = 'w' if i == 0 else 'a'
				header = (i == 0)
				result_df.to_csv(out_csv, index=False, mode=mode, header=header)
				logging.info(f"Saved {len(result_df)} rows from chunk {i+1} to {out_csv}")
			else:
				logging.info(f"No crime-related rows found in chunk {i+1}")

	logging.info("Zero-shot step completed.")


# ------------------------------
# Step 4: Sentence chunking
# ------------------------------

def ensure_spacy_model(model_name: str, auto_download: bool = True):
	try:
		return spacy.load(model_name)
	except OSError:
		if auto_download:
			logging.info(f"spaCy model '{model_name}' not found. Downloading...")
			spacy_download(model_name)
			return spacy.load(model_name)
		raise


def step_process(cfg: Dict[str, Any], in_csv: str, out_csv: str) -> None:
	if os.path.exists(out_csv) and not cfg.get("general", {}).get("force", False):
		logging.info(f"Processed output exists at {out_csv}; skipping.")
		return

	proc_cfg = cfg["process"]
	io_text_col = cfg["io"]["text_column"]
	chunksize_csv = int(proc_cfg.get("chunksize", 100))
	max_tokens = int(proc_cfg.get("max_tokens", 250))
	min_tokens = int(proc_cfg.get("min_tokens", 80))
	output_column = proc_cfg.get("output_column", "sentence1")

	logging.info("Loading spaCy model and tokenizer for token counting...")
	nlp = ensure_spacy_model(proc_cfg.get("spacy_model", "it_core_news_lg"), auto_download=bool(proc_cfg.get("auto_download_spacy_model", True)))
	tokenizer = AutoTokenizer.from_pretrained(proc_cfg.get("tokenizer_name", "intfloat/multilingual-e5-large"))

	def count_tokens(text: str) -> int:
		return len(tokenizer.encode(text))

	def split_paragraph_with_token_limit(paragraph: str) -> List[str]:
		try:
			doc = nlp(paragraph.strip())
			sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
			result_chunks: List[str] = []
			current_chunk: List[str] = []
			current_token_count = 0

			for sent in sentences:
				sent_token_count = count_tokens(sent)
				if current_token_count + sent_token_count > max_tokens:
					if current_chunk:
						chunk_text = " ".join(current_chunk)
						if count_tokens(chunk_text) >= min_tokens:
							result_chunks.append(chunk_text)
					current_chunk = [sent]
					current_token_count = sent_token_count
				else:
					current_chunk.append(sent)
					current_token_count += sent_token_count

			if current_chunk:
				chunk_text = " ".join(current_chunk)
				if count_tokens(chunk_text) >= min_tokens:
					result_chunks.append(chunk_text)

			return result_chunks
		except Exception as e:
			logging.warning(f"Error processing paragraph: {e}")
			return []

	# Prepare output file with header
	with open(out_csv, 'w', encoding='utf-8') as f:
		f.write(f"{output_column}\n")

	logging.info(f"Processing input file into sentence chunks: {in_csv}")
	chunk_iter = pd.read_csv(in_csv, usecols=[io_text_col], chunksize=chunksize_csv)

	for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
		output_rows: List[str] = []
		for _, row in chunk.iterrows():
			paragraph = str(row[io_text_col])
			chunks = split_paragraph_with_token_limit(paragraph)
			output_rows.extend(chunks)

		df_out = pd.DataFrame(output_rows, columns=[output_column])
		df_out.to_csv(out_csv, mode='a', index=False, header=False)
		logging.info(f"Processed chunk {i+1}, saved {len(output_rows)} samples.")

	logging.info("Processing completed.")


# ------------------------------
# Orchestrator
# ------------------------------

def run_pipeline(config_path: str, steps: List[str] = None) -> None:
	if steps is None:
		steps = ["download", "keyword_filter", "zero_shot", "process"]

	cfg = load_config(config_path)
	paths = build_paths(cfg)

	logging.info("Using outputs directory: %s", Path(cfg["io"]["base_dir"]))

	if "download" in steps:
		step_download(cfg, paths["download_csv"])
	if "keyword_filter" in steps:
		step_keyword_filter(cfg, paths["download_csv"], paths["filtered_csv"])
	if "zero_shot" in steps:
		step_zero_shot(cfg, paths["filtered_csv"], paths["zero_shot_csv"])
	if "process" in steps:
		step_process(cfg, paths["zero_shot_csv"], paths["processed_csv"])


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Preprocessing pipeline")
	parser.add_argument("--config", type=str, default="/home/haiderali/Desktop/Italian Language Embedder/config.yaml", help="Path to config.yaml")
	parser.add_argument("--steps", type=str, nargs="*", default=["download", "keyword_filter", "zero_shot", "process"], help="Subset of steps to run: download keyword_filter zero_shot process")
	parser.add_argument("--force", action="store_true", help="Force re-run steps even if outputs exist")
	parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

	args = parser.parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

	# Allow overriding force from CLI
	if args.force:
		# Load, modify, and pass transient config
		cfg = load_config(args.config)
		cfg.setdefault("general", {})["force"] = True
		# Run with modified cfg without writing back
		# We call internal functions to avoid re-reading from disk
		paths = build_paths(cfg)
		logging.info("Force mode enabled. Re-running steps: %s", args.steps)
		if "download" in args.steps:
			step_download(cfg, paths["download_csv"])
		if "keyword_filter" in args.steps:
			step_keyword_filter(cfg, paths["download_csv"], paths["filtered_csv"])
		if "zero_shot" in args.steps:
			step_zero_shot(cfg, paths["filtered_csv"], paths["zero_shot_csv"])
		if "process" in args.steps:
			step_process(cfg, paths["zero_shot_csv"], paths["processed_csv"])
	else:
		run_pipeline(args.config, args.steps)
