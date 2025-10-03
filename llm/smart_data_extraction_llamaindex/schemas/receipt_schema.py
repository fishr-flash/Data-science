"""Receipt-specific Pydantic schemas for document extraction."""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


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
