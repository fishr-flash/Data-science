# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "llama-index",
#     "llama-index-program-openai",
#     "llama-parse",
#     "python-dotenv",
#     "rapidfuzz",
# ]
# ///
import os
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from llama_parse import LlamaParse
from PIL import Image
from pydantic import BaseModel, Field
from rapidfuzz import fuzz


def configure_settings() -> None:
	"""Load environment variables and configure the default LLM."""
	load_dotenv(override=True)

	openai_key = os.environ.get("OPENAI_API_KEY")
	Settings.llm = OpenAI(api_key=openai_key, model="gpt-4o-mini", temperature=0)
	Settings.context_window = 8000


def scale_image(image_path: Path, output_dir: Path, scale_factor: int = 3) -> Path:
	"""Scale up an image using high-quality resampling.

	Args:
	    image_path: Path to the original image
	    output_dir: Directory to save the scaled image
	    scale_factor: Factor to scale up the image (default: 3x)

	Returns:
	    Path to the scaled image
	"""
	# Load the image
	img = Image.open(image_path)

	# Scale up the image using high-quality resampling
	new_size = (img.width * scale_factor, img.height * scale_factor)
	img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

	# Save to output directory with same filename
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / image_path.name
	img_resized.save(output_path, quality=95)

	return output_path


class ReceiptItem(BaseModel):
	"""Line item extracted from a receipt."""

	description: str = Field(description="Item name exactly as shown on the receipt")
	quantity: int = Field(default=1, ge=1, description="Integer quantity of the item")
	unit_price: Optional[float] = Field(
		default=None, ge=0, description="Price per unit in the receipt currency"
	)
	discount_amount: float = Field(
		default=0.0, ge=0, description="Discount applied to this line item"
	)


class Receipt(BaseModel):
	"""Structured receipt fields extracted from OCR."""

	company: str = Field(description="Business or merchant name")
	purchase_date: Optional[date] = Field(
		default=None, description="Date in YYYY-MM-DD format"
	)
	address: Optional[str] = Field(default=None, description="Address of the business")
	total: float = Field(description="Final charged amount")
	items: List[ReceiptItem] = Field(default_factory=list)


def extract_documents(paths: List[str], prompt: str, id_column: str) -> List[dict]:
	"""Extract structured data from documents using LlamaParse and LLM.

	Args:
	    paths: List of document file paths
	    prompt: Extraction prompt template
	    id_column: ID column to identify the document

	Returns:
	    List of dictionaries with document_id and extracted data
	"""
	results: List[dict] = []

	parser = LlamaParse(
		api_key=os.environ["LLAMA_CLOUD_API_KEY"],
		result_type="markdown",
		num_workers=4,
		language="en",
		skip_diagonal_text=True,
	)

	documents = parser.load_data(paths)

	program = OpenAIPydanticProgram.from_defaults(
		output_cls=Receipt,
		llm=Settings.llm,
		prompt_template_str=prompt,
	)

	for path, doc in zip(paths, documents, strict=False):
		document_id = Path(path).stem
		parsed_document = program(context_str=doc.text)
		results.append(
			{
				id_column: document_id,
				"data": parsed_document,
			}
		)
	return results


def transform_receipt_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply standard transformations to receipt DataFrame columns.

	Transforms:
	- company: Convert to uppercase
	- total: Convert to numeric
	- purchase_date: Convert to date
	"""
	df = df.copy()

	df["company"] = df["company"].str.upper()

	df["total"] = pd.to_numeric(df["total"], errors="coerce")

	df["purchase_date"] = pd.to_datetime(
		df["purchase_date"], errors="coerce", dayfirst=True
	).dt.date

	return df


def create_extracted_df(records: List[dict], id_column: str) -> pd.DataFrame:
	df = pd.DataFrame(
		[
			{
				id_column: record[id_column],
				"company": record["data"].company,
				"total": record["data"].total,
				"purchase_date": record["data"].purchase_date,
			}
			for record in records
		]
	)
	return transform_receipt_columns(df)


def normalize_date(value: str) -> str:
	value = (value or "").strip()
	if not value:
		return value
	value = value.replace("-", "/")
	parts = value.split("/")
	if len(parts[-1]) == 2:
		parts[-1] = f"20{parts[-1]}"
	return "/".join(parts)


def create_ground_truth_df(label_paths: List[str], id_column: str) -> pd.DataFrame:
	"""Create ground truth DataFrame from label JSON files."""
	records = []
	for path in label_paths:
		payload = pd.read_json(Path(path), typ="series").to_dict()
		records.append(
			{
				id_column: Path(path).stem,
				"company": payload.get("company", ""),
				"total": payload.get("total", ""),
				"purchase_date": normalize_date(payload.get("date", "")),
			}
		)

	df = pd.DataFrame(records)
	return transform_receipt_columns(df)


def fuzzy_match_score(text1: str, text2: str) -> int:
	"""Calculate fuzzy match score between two strings.

	Args:
	    text1: First string to compare
	    text2: Second string to compare

	Returns:
	    Similarity score between 0 and 100
	"""
	return fuzz.token_set_ratio(str(text1), str(text2))


def compare_receipts(
	extracted_df: pd.DataFrame,
	ground_truth_df: pd.DataFrame,
	id_column: str,
	fuzzy_match_cols: List[str],
	exact_match_cols: List[str],
	fuzzy_threshold: int = 80,
) -> pd.DataFrame:
	"""Compare extracted and ground truth data with explicit column specifications.

	Args:
	    extracted_df: DataFrame with extracted data
	    ground_truth_df: DataFrame with ground truth data
	    id_column: Column to join on
	    fuzzy_match_cols: Columns to compare using fuzzy matching
	    exact_match_cols: Columns to compare using exact matching
	    fuzzy_threshold: Similarity threshold for fuzzy matching (default: 80)
	"""
	comparison_df = extracted_df.merge(
		ground_truth_df,
		on=id_column,
		how="inner",
		suffixes=("_extracted", "_truth"),
	)

	# Fuzzy matching
	for col in fuzzy_match_cols:
		extracted_col = f"{col}_extracted"
		truth_col = f"{col}_truth"
		comparison_df[f"{col}_score"] = comparison_df.apply(
			lambda row, ec=extracted_col, tc=truth_col: fuzzy_match_score(row[ec], row[tc]),
			axis=1,
		)
		comparison_df[f"{col}_match"] = comparison_df[f"{col}_score"] >= fuzzy_threshold

	# Exact matching
	for col in exact_match_cols:
		extracted_col = f"{col}_extracted"
		truth_col = f"{col}_truth"
		comparison_df[f"{col}_match"] = (
			comparison_df[extracted_col] == comparison_df[truth_col]
		)

	return comparison_df


def get_mismatch_rows(comparison_df: pd.DataFrame) -> pd.DataFrame:
	"""Get mismatched rows, excluding match indicator columns."""
	# Extract match columns and data columns
	match_columns = [col for col in comparison_df.columns if col.endswith("_match")]
	data_columns = [col for col in comparison_df.columns if col not in match_columns]

	# Check for rows where not all matches are True
	has_mismatch = comparison_df[match_columns].all(axis=1).eq(False)

	return comparison_df[has_mismatch][data_columns]


def main(
	receipt_paths: List[str],
	label_paths: List[str],
	preprocess: bool = True,
	output_dir: Path = Path("data/SROIE2019/train/img_adjusted"),
	id_column: str = "receipt_id",
) -> None:
	configure_settings()

	# Preprocess images if requested
	if preprocess:
		print("Preprocessing receipt images...")
		receipt_paths_to_parse = [
			scale_image(Path(p), output_dir, scale_factor=3) for p in receipt_paths
		]
	else:
		receipt_paths_to_parse = receipt_paths

	prompt = """
    You are extracting structured data from a receipt.
    Use the provided text to populate the Receipt model.
    Interpret every receipt date as day-first.
    If a field is missing, return null.

    {context_str}
    """

	structured_receipts = extract_documents(receipt_paths_to_parse, prompt, id_column)

	extracted_df = create_extracted_df(structured_receipts, id_column)
	ground_truth_df = create_ground_truth_df(label_paths, id_column)

	comparison_df = compare_receipts(
		extracted_df,
		ground_truth_df,
		id_column,
		fuzzy_match_cols=["company"],
		exact_match_cols=["total", "purchase_date"],
	)
	mismatch_df = get_mismatch_rows(comparison_df)

	if mismatch_df.empty:
		print("All receipts matched the ground truth.")
	else:
		print("Mismatched receipts:")
		print(mismatch_df)


if __name__ == "__main__":
	# Default paths
	receipt_dir = Path("data/SROIE2019/train/img")
	label_dir = Path("data/SROIE2019/train/entities")
	adjusted_receipt_dir = Path("data/SROIE2019/train/img_adjusted")

	# Default number of receipts
	num_receipts = 10
	receipt_paths = sorted(receipt_dir.glob("*.jpg"))[:num_receipts]
	label_paths = sorted(label_dir.glob("*.txt"))[:num_receipts]

	# Run the pipeline
	main(receipt_paths, label_paths, preprocess=True, output_dir=adjusted_receipt_dir)
