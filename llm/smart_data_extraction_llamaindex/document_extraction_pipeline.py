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
"""Generic document extraction pipeline for any document type."""

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


def process_images(
	image_paths: List[str],
	output_dir: Path,
	transform_image_fn: Callable[[Image.Image], Image.Image]
) -> List[str]:
	"""Process images by applying a transformation function.

	Generic infrastructure that loads images, applies transformation, and saves them.

	Args:
	    image_paths: List of paths to images
	    output_dir: Directory to save processed images
	    transform_image_fn: Function that takes PIL Image and returns transformed PIL Image

	Returns:
	    List of paths to processed images
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	processed_paths = []

	for path in image_paths:
		# Load image
		img = Image.open(path)

		# Apply transformation
		img_transformed = transform_image_fn(img)

		# Save transformed image
		output_path = output_dir / Path(path).name
		img_transformed.save(output_path, quality=95)
		processed_paths.append(str(output_path))

	return processed_paths


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
	data_transformer: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> pd.DataFrame:
	"""Create DataFrame from extracted records.

	Args:
	    records: List of extraction results with id and data
	    id_column: Column name for document IDs
	    fields: List of field names to extract from the Pydantic model
	    data_transformer: Optional function to transform the DataFrame

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

	if data_transformer:
		df = data_transformer(df)

	return df


def extract_structured_data(
	image_paths: List[str],
	output_cls: Type[BaseModel],
	prompt: str,
	id_column: str = "document_id",
	fields: Optional[List[str]] = None,
	image_transform_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
	image_output_dir: Optional[Path] = None,
	data_transformer: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
	"""Extract structured data from documents using a generic pipeline.

	Args:
	    image_paths: Paths to images/documents to process
	    output_cls: Pydantic model class defining the extraction schema
	    prompt: Extraction prompt template (must include {context_str})
	    id_column: Column name for document identifiers
	    fields: List of field names to extract (if None, uses all model fields)
	    image_transform_fn: Optional function to transform individual images (takes PIL Image, returns PIL Image)
	    image_output_dir: Directory to save transformed images (required if image_transform_fn provided)
	    data_transformer: Optional function to transform the extracted DataFrame

	Returns:
	    DataFrame with extracted data
	"""
	configure_settings()

	# Infer fields from model if not provided
	if fields is None:
		fields = list(output_cls.model_fields.keys())

	# Process images if transformation function provided
	if image_transform_fn:
		if image_output_dir is None:
			raise ValueError("image_output_dir must be provided when image_transform_fn is specified")
		print("Processing images...")
		paths_to_parse = process_images(image_paths, image_output_dir, image_transform_fn)
	else:
		paths_to_parse = image_paths

	# Extract documents
	structured_data = extract_documents(paths_to_parse, prompt, id_column, output_cls)

	# Create extracted DataFrame
	extracted_df = create_extracted_df(structured_data, id_column, fields, data_transformer)

	return extracted_df
