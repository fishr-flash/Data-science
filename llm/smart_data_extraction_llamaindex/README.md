# Generic Document Extraction Pipeline

A flexible, schema-driven pipeline for extracting structured data from any type of document or image using LlamaParse and OpenAI.

## Features

- **Dynamic Schema Support**: Use any Pydantic model to define your extraction schema
- **Optional Preprocessing**: Scale and optimize images before extraction
- **Flexible Transformations**: Apply custom transformation functions to extracted data
- **Extensible**: Easy to adapt for receipts, invoices, forms, IDs, or any document type

## Quick Start

### 1. Define Your Schema

Create a Pydantic model for your document type:

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    vendor_name: str = Field(description="Vendor name")
    total_amount: float = Field(description="Total amount")
```

### 2. Run Extraction

```python
from extract_receipts_pipeline import main

result_df = main(
    image_paths=["invoice1.pdf", "invoice2.pdf"],
    output_cls=Invoice,
    prompt="Extract invoice data from: {context_str}",
    id_column="invoice_id",
)
```

## Usage Examples

### Basic Extraction (No Ground Truth)

```python
from schemas.receipt_schema import Receipt
from extract_receipts_pipeline import main

result = main(
    image_paths=["receipt1.jpg"],
    output_cls=Receipt,
    prompt="Extract receipt data: {context_str}",
)
```

### With Preprocessing

```python
from pathlib import Path

result = main(
    image_paths=["low_res.jpg"],
    output_cls=Receipt,
    prompt="Extract data: {context_str}",
    preprocess=True,
    output_dir=Path("processed_images"),
    scale_factor=3,
)
```

### With Custom Transformations

```python
import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df["vendor"] = df["vendor"].str.upper()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df

result = main(
    image_paths=["invoice.pdf"],
    output_cls=Invoice,
    prompt="Extract: {context_str}",
    transform_fn=transform_data,
)
```

## Parameters

### Required
- `image_paths`: List of document/image paths
- `output_cls`: Pydantic model class for extraction
- `prompt`: Extraction prompt template (must include `{context_str}`)

### Optional
- `id_column`: Document ID column name (default: "document_id")
- `fields`: Fields to extract (default: all model fields)
- `preprocess`: Enable image preprocessing (default: False)
- `output_dir`: Directory for preprocessed images
- `scale_factor`: Image scaling factor (default: 3)
- `transform_fn`: Custom transformation function

## File Structure

```
llm/smart_data_extraction_llamaindex/
├── extract_receipts_pipeline.py   # Main pipeline
├── schemas/
│   ├── __init__.py
│   └── receipt_schema.py          # Receipt schema example
├── example_invoice.py             # Invoice extraction example
└── README.md                      # This file
```

## Custom Schema Examples

See:
- `schemas/receipt_schema.py` - Receipt extraction
- `example_invoice.py` - Invoice extraction example

Create your own schemas in the `schemas/` directory!
