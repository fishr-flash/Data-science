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
"""Async document extraction pipeline with parallel processing for improved performance."""

import asyncio
import os
from pathlib import Path
from typing import Callable, List, Optional, Type

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
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


def _process_single_image_sync(
	path: str,
	output_dir: Path,
	transform_image_fn: Callable[[Image.Image], Image.Image]
) -> str:
	"""Process a single image synchronously (CPU-bound operation).

	Args:
	    path: Path to the image
	    output_dir: Directory to save processed image
	    transform_image_fn: Function to transform the image

	Returns:
	    Path to the processed image
	"""
	img = Image.open(path)
	img_transformed = transform_image_fn(img)
	output_path = output_dir / Path(path).name
	img_transformed.save(output_path, quality=95)
	return str(output_path)


async def _process_single_image_async(
	path: str,
	output_dir: Path,
	transform_image_fn: Callable[[Image.Image], Image.Image]
) -> str:
	"""Process a single image asynchronously (wraps sync operations).

	Args:
	    path: Path to the image
	    output_dir: Directory to save processed image
	    transform_image_fn: Function to transform the image

	Returns:
	    Path to the processed image
	"""
	loop = asyncio.get_event_loop()
	return await loop.run_in_executor(
		None,
		_process_single_image_sync,
		path,
		output_dir,
		transform_image_fn
	)


async def process_images_async(
	image_paths: List[str],
	output_dir: Path,
	transform_image_fn: Callable[[Image.Image], Image.Image],
	num_workers: int = 4
) -> List[str]:
	"""Process images asynchronously by applying a transformation function.

	Args:
	    image_paths: List of paths to images
	    output_dir: Directory to save processed images
	    transform_image_fn: Function that takes PIL Image and returns transformed PIL Image
	    num_workers: Number of parallel workers for processing

	Returns:
	    List of paths to processed images
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	# Process images concurrently
	tasks = [
		_process_single_image_async(path, output_dir, transform_image_fn)
		for path in image_paths
	]
	processed_paths = await run_jobs(tasks, show_progress=True, workers=num_workers)

	return processed_paths


def _extract_single_document_sync(
	path: str,
	doc,
	id_column: str,
	program: OpenAIPydanticProgram
) -> dict:
	"""Extract structured data from a single document synchronously.

	Args:
	    path: Path to the document
	    doc: Parsed document object
	    id_column: ID column name
	    program: Pydantic program for extraction

	Returns:
	    Dictionary with document_id and extracted data
	"""
	document_id = Path(path).stem
	parsed_document = program(context_str=doc.text)

	return {
		id_column: document_id,
		"data": parsed_document,
	}


async def _extract_single_document_async(
	path: str,
	doc,
	id_column: str,
	program: OpenAIPydanticProgram
) -> dict:
	"""Extract structured data from a single document asynchronously.

	Args:
	    path: Path to the document
	    doc: Parsed document object
	    id_column: ID column name
	    program: Pydantic program for extraction

	Returns:
	    Dictionary with document_id and extracted data
	"""
	loop = asyncio.get_event_loop()
	return await loop.run_in_executor(
		None,
		_extract_single_document_sync,
		path,
		doc,
		id_column,
		program
	)


async def extract_documents_async(
	paths: List[str],
	prompt: str,
	id_column: str,
	output_cls: Type[BaseModel],
	num_workers: int = 4
) -> List[dict]:
	"""Extract structured data from documents asynchronously using LlamaParse and LLM.

	Args:
	    paths: List of document file paths
	    prompt: Extraction prompt template
	    id_column: ID column to identify the document
	    output_cls: Pydantic model class for structured output
	    num_workers: Number of parallel workers for LLM extraction

	Returns:
	    List of dictionaries with document_id and extracted data
	"""
	# Initialize parser
	parser = LlamaParse(
		api_key=os.environ["LLAMA_CLOUD_API_KEY"],
		result_type="markdown",
		num_workers=num_workers,  # LlamaParse handles parallel parsing
		language="en",
		skip_diagonal_text=True,
	)

	# Parse documents (this is already parallel via num_workers)
	documents = parser.load_data(paths)

	# Initialize extraction program
	program = OpenAIPydanticProgram.from_defaults(
		output_cls=output_cls,
		llm=Settings.llm,
		prompt_template_str=prompt,
	)

	# Process all documents concurrently
	tasks = [
		_extract_single_document_async(path, doc, id_column, program)
		for path, doc in zip(paths, documents, strict=False)
	]
	results = await run_jobs(tasks, show_progress=True, workers=num_workers)

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


async def extract_structured_data_async(
	image_paths: List[str],
	output_cls: Type[BaseModel],
	prompt: str,
	id_column: str = "document_id",
	fields: Optional[List[str]] = None,
	image_transform_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
	image_output_dir: Optional[Path] = None,
	data_transformer: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
	num_workers: int = 4,
) -> pd.DataFrame:
	"""Extract structured data from documents asynchronously using a generic pipeline.

	This async version provides significant performance improvements through:
	- Parallel image processing
	- Concurrent LLM extraction calls
	- Non-blocking I/O operations

	Args:
	    image_paths: Paths to images/documents to process
	    output_cls: Pydantic model class defining the extraction schema
	    prompt: Extraction prompt template (must include {context_str})
	    id_column: Column name for document identifiers
	    fields: List of field names to extract (if None, uses all model fields)
	    image_transform_fn: Optional function to transform individual images
	    image_output_dir: Directory to save transformed images
	    data_transformer: Optional function to transform the extracted DataFrame
	    num_workers: Number of parallel workers for processing (default: 4)

	Returns:
	    DataFrame with extracted data
	"""
	configure_settings()

	# Infer fields from model if not provided
	if fields is None:
		fields = list(output_cls.model_fields.keys())

	# Process images asynchronously if transformation function provided
	if image_transform_fn:
		if image_output_dir is None:
			raise ValueError("image_output_dir must be provided when image_transform_fn is specified")
		print("Processing images asynchronously...")
		paths_to_parse = await process_images_async(
			image_paths, image_output_dir, image_transform_fn, num_workers
		)
	else:
		paths_to_parse = image_paths

	# Extract documents asynchronously
	print(f"Extracting documents with {num_workers} parallel workers...")
	structured_data = await extract_documents_async(
		paths_to_parse, prompt, id_column, output_cls, num_workers
	)

	# Create extracted DataFrame
	extracted_df = create_extracted_df(structured_data, id_column, fields, data_transformer)

	return extracted_df
