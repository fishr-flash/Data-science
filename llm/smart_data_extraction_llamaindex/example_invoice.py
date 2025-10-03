"""Example: Using the generic extraction pipeline with a custom invoice schema."""

from typing import Optional

import pandas as pd

# Import the generic extraction pipeline
from extract_receipts_pipeline import main
from pydantic import BaseModel, Field


# Define custom schema for invoices
class Invoice(BaseModel):
	"""Invoice extraction schema."""

	invoice_number: str = Field(description="Invoice number or ID")
	vendor_name: str = Field(description="Vendor or supplier name")
	invoice_date: Optional[str] = Field(default=None, description="Invoice date")
	total_amount: float = Field(description="Total invoice amount")
	tax_amount: Optional[float] = Field(default=None, description="Tax amount if present")


# Optional: Define transformation function
def transform_invoice_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Transform invoice data."""
	df = df.copy()
	df["vendor_name"] = df["vendor_name"].str.upper()
	df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
	df["tax_amount"] = pd.to_numeric(df["tax_amount"], errors="coerce")
	return df


# Define extraction prompt
INVOICE_PROMPT = """
You are extracting structured data from an invoice document.
Use the provided text to populate the Invoice model accurately.
If a field is not present in the document, return null.

{context_str}
"""


if __name__ == "__main__":
	# Example usage - replace with your actual invoice paths
	invoice_paths = [
		"path/to/invoice1.pdf",
		"path/to/invoice2.pdf",
	]

	# Run extraction
	result_df = main(
		image_paths=invoice_paths,
		output_cls=Invoice,
		prompt=INVOICE_PROMPT,
		id_column="invoice_id",
		fields=["invoice_number", "vendor_name", "invoice_date", "total_amount", "tax_amount"],
		transform_fn=transform_invoice_data,
	)

	print("\nExtracted Invoices:")
	print(result_df)

	# Save results
	result_df.to_csv("extracted_invoices.csv", index=False)
	print("\nResults saved to extracted_invoices.csv")
