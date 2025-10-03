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
"""Receipt extraction pipeline with schema and processing logic."""

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
from document_extraction_pipeline import extract_structured_data
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


# Receipt-specific transformations
def transform_receipt_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply receipt-specific transformations."""
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


if __name__ == "__main__":
	# Default paths
	receipt_dir = Path("data/SROIE2019/train/img")
	adjusted_receipt_dir = Path("data/SROIE2019/train/img_adjusted")

	# Default number of receipts
	num_receipts = 10
	receipt_paths = sorted(receipt_dir.glob("*.jpg"))[:num_receipts]

	# Run the pipeline
	result_df = extract_structured_data(
		image_paths=receipt_paths,
		output_cls=Receipt,
		prompt=RECEIPT_PROMPT,
		id_column="receipt_id",
		fields=["company", "total", "purchase_date"],
		preprocess=True,
		output_dir=adjusted_receipt_dir,
		scale_factor=3,
		transform_fn=transform_receipt_columns,
	)

	print("\nExtraction complete!")
	print(result_df)
