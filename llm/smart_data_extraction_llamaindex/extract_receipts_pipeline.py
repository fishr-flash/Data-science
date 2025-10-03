# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "llama-index",
#     "llama-index-program-openai",
#     "llama-parse",
#     "python-dotenv",
#     "pandas",
#     "pillow",
# ]
# ///
import os
from pathlib import Path
from typing import Callable, List, Optional, Type

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from llama_parse import LlamaParse
from PIL import Image
from pydantic import BaseModel


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


def extract_documents(
	paths: List[str],
	prompt: str,
	id_column: str,
	output_cls: Type[BaseModel]
) -> List[dict]:
	"""Extract structured data from documents using LlamaParse and LLM.

	Args:
	    paths: List of document file paths
	    prompt: Extraction prompt template
	    id_column: ID column to identify the document
	    output_cls: Pydantic model class for structured output

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
		output_cls=output_cls,
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


def create_extracted_df(
	records: List[dict],
	id_column: str,
	fields: List[str],
	transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> pd.DataFrame:
	"""Create DataFrame from extracted records.

	Args:
	    records: List of extraction results with id and data
	    id_column: Column name for document IDs
	    fields: List of field names to extract from the Pydantic model
	    transform_fn: Optional function to transform the DataFrame

	Returns:
	    DataFrame with extracted fields
	"""
	df = pd.DataFrame(
		[
			{
				id_column: record[id_column],
				**{field: getattr(record["data"], field) for field in fields}
			}
			for record in records
		]
	)

	if transform_fn:
		df = transform_fn(df)

	return df


def main(
	image_paths: List[str],
	output_cls: Type[BaseModel],
	prompt: str,
	id_column: str = "document_id",
	fields: Optional[List[str]] = None,
	preprocess: bool = False,
	output_dir: Optional[Path] = None,
	scale_factor: int = 3,
	transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
	"""Generic document extraction pipeline.

	Args:
	    image_paths: Paths to images/documents to process
	    output_cls: Pydantic model class defining the extraction schema
	    prompt: Extraction prompt template (must include {context_str})
	    id_column: Column name for document identifiers
	    fields: List of field names to extract (if None, uses all model fields)
	    preprocess: Whether to scale/preprocess images
	    output_dir: Directory for preprocessed images
	    scale_factor: Image scaling factor if preprocessing
	    transform_fn: Optional transformation function for DataFrames

	Returns:
	    DataFrame with extracted data
	"""
	configure_settings()

	# Infer fields from model if not provided
	if fields is None:
		fields = list(output_cls.model_fields.keys())

	# Preprocess images if requested
	if preprocess:
		if output_dir is None:
			raise ValueError("output_dir must be provided when preprocess=True")
		print("Preprocessing images...")
		paths_to_parse = [
			scale_image(Path(p), output_dir, scale_factor=scale_factor)
			for p in image_paths
		]
	else:
		paths_to_parse = image_paths

	# Extract documents
	structured_data = extract_documents(paths_to_parse, prompt, id_column, output_cls)

	# Create extracted DataFrame
	extracted_df = create_extracted_df(structured_data, id_column, fields, transform_fn)

	return extracted_df


if __name__ == "__main__":
	# Example: Receipt extraction
	from schemas.receipt_schema import Receipt

	def transform_receipt_columns(df: pd.DataFrame) -> pd.DataFrame:
		"""Apply receipt-specific transformations."""
		df = df.copy()
		df["company"] = df["company"].str.upper()
		df["total"] = pd.to_numeric(df["total"], errors="coerce")
		df["purchase_date"] = pd.to_datetime(
			df["purchase_date"], errors="coerce", dayfirst=True
		).dt.date
		return df

	# Default paths
	receipt_dir = Path("data/SROIE2019/train/img")
	adjusted_receipt_dir = Path("data/SROIE2019/train/img_adjusted")

	# Default number of receipts
	num_receipts = 10
	receipt_paths = sorted(receipt_dir.glob("*.jpg"))[:num_receipts]

	# Receipt extraction prompt
	prompt = """
    You are extracting structured data from a receipt.
    Use the provided text to populate the Receipt model.
    Interpret every receipt date as day-first.
    If a field is missing, return null.

    {context_str}
    """

	# Run the pipeline
	result_df = main(
		image_paths=receipt_paths,
		output_cls=Receipt,
		prompt=prompt,
		id_column="receipt_id",
		fields=["company", "total", "purchase_date"],
		preprocess=True,
		output_dir=adjusted_receipt_dir,
		scale_factor=3,
		transform_fn=transform_receipt_columns,
	)

	print("\nExtraction complete!")
	print(result_df)
