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
"""Async receipt extraction pipeline with parallel processing for better performance."""

import asyncio
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
from async_document_extraction_pipeline import extract_structured_data_async
from PIL import Image
from pydantic import BaseModel, Field


# Receipt-specific schemas
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


# Receipt-specific image transformation
def scale_receipt_image(img: Image.Image, scale_factor: int = 3) -> Image.Image:
	"""Scale up a receipt image for better OCR.

	Args:
	    img: PIL Image object
	    scale_factor: Factor to scale up the image (default: 3x)

	Returns:
	    Transformed PIL Image
	"""
	new_size = (img.width * scale_factor, img.height * scale_factor)
	return img.resize(new_size, Image.Resampling.LANCZOS)


# Receipt-specific data transformations
def transform_receipt_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Transform extracted receipt data (normalize text, convert types).

	Args:
	    df: DataFrame with extracted receipt data

	Returns:
	    Transformed DataFrame
	"""
	df = df.copy()
	df["company"] = df["company"].str.upper()
	df["total"] = pd.to_numeric(df["total"], errors="coerce")
	df["purchase_date"] = pd.to_datetime(
		df["purchase_date"], errors="coerce", dayfirst=True
	).dt.date
	return df


# Receipt extraction prompt
RECEIPT_PROMPT = """
You are extracting structured data from a receipt.
Use the provided text to populate the Receipt model.
Interpret every receipt date as day-first.
If a field is missing, return null.

{context_str}
"""


async def extract_receipts(
	receipt_paths: List[Path], output_dir: Path, num_workers: int = 4
) -> pd.DataFrame:
	"""Extract receipt data asynchronously with parallel processing.

	This async version provides significant performance improvements for batch processing:
	- Parallel image preprocessing
	- Concurrent LLM extraction calls
	- Non-blocking I/O operations

	Args:
	    receipt_paths: List of paths to receipt images
	    output_dir: Directory to save processed images
	    num_workers: Number of parallel workers (default: 4)

	Returns:
	    DataFrame with extracted receipt data

	Example:
	    >>> import asyncio
	    >>> from pathlib import Path
	    >>>
	    >>> async def main():
	    ...     paths = [Path("receipt1.jpg"), Path("receipt2.jpg")]
	    ...     result = await extract_receipts(
	    ...         receipt_paths=paths,
	    ...         output_dir=Path("processed"),
	    ...         num_workers=4
	    ...     )
	    ...     print(result)
	    >>>
	    >>> asyncio.run(main())
	"""
	print(f"Starting async extraction of {len(receipt_paths)} receipts...")
	print(f"Using {num_workers} parallel workers")

	result_df = await extract_structured_data_async(
		image_paths=[str(p) for p in receipt_paths],
		output_cls=Receipt,
		prompt=RECEIPT_PROMPT,
		id_column="receipt_id",
		fields=["company", "total", "purchase_date"],
		image_transform_fn=scale_receipt_image,
		image_output_dir=output_dir,
		data_transformer=transform_receipt_data,
		num_workers=num_workers,
	)

	print(f"✓ Extraction complete! Processed {len(result_df)} receipts")
	return result_df


async def main():
	"""Main async function to run the receipt extraction pipeline."""
	# Default paths
	receipt_dir = Path("data/SROIE2019/train/img")
	adjusted_receipt_dir = Path("data/SROIE2019/train/img_adjusted")

	# Get receipt paths
	num_receipts = 10
	receipt_paths = sorted(receipt_dir.glob("*.jpg"))[:num_receipts]

	if not receipt_paths:
		print(f"No receipts found in {receipt_dir}")
		print("Please ensure the SROIE2019 dataset is available.")
		return

	# Run async extraction
	result_df = await extract_receipts(
		receipt_paths=receipt_paths,
		output_dir=adjusted_receipt_dir,
		num_workers=4,  # Process 4 receipts in parallel
	)

	# Display results
	print("\n" + "=" * 80)
	print("EXTRACTED RECEIPT DATA")
	print("=" * 80)
	print(result_df.to_string())

	# Save to CSV
	output_csv = Path("reports/receipts_async.csv")
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(output_csv, index=False)
	print(f"\n✓ Results saved to {output_csv}")


if __name__ == "__main__":
	asyncio.run(main())
