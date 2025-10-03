# Document Extraction Pipeline

A flexible, schema-driven pipeline for extracting structured data from any type of document or image using LlamaParse and OpenAI.

## Features

- **Dynamic Schema Support**: Use any Pydantic model to define your extraction schema
- **Optional Preprocessing**: Scale and optimize images before extraction
- **Flexible Transformations**: Apply custom transformation functions to extracted data
- **Extensible**: Easy to adapt for receipts, invoices, forms, IDs, or any document type

## File Structure

```
llm/smart_data_extraction_llamaindex/
├── document_extraction_pipeline.py   # Generic pipeline (reusable)
├── extract_receipts_pipeline.py      # Receipt-specific (schema + logic + example)
└── README.md                          # This file
```

## Quick Start

### Option 1: Use the Receipt Pipeline

Run the ready-to-use receipt extraction pipeline:

```bash
uv run extract_receipts_pipeline.py
```

The receipt pipeline includes:
- `Receipt` and `ReceiptItem` Pydantic schemas
- Receipt-specific data transformations
- Pre-configured extraction prompt
- Example usage in `__main__` block

### Option 2: Create Your Own Pipeline

Import the generic pipeline and create a custom extractor:

```python
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from document_extraction_pipeline import extract_structured_data


# 1. Define your schema
class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    vendor_name: str = Field(description="Vendor name")
    invoice_date: Optional[date] = Field(default=None)
    total_amount: float = Field(description="Total amount")


# 2. Optional: Define transformations
def transform_invoice_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vendor_name"] = df["vendor_name"].str.upper()
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    return df


# 3. Define extraction prompt
INVOICE_PROMPT = """
Extract invoice data from the following document.
If a field is missing, return null.

{context_str}
"""


# 4. Run extraction
if __name__ == "__main__":
    invoice_paths = ["invoice1.pdf", "invoice2.pdf"]

    result_df = extract_structured_data(
        image_paths=invoice_paths,
        output_cls=Invoice,
        prompt=INVOICE_PROMPT,
        id_column="invoice_id",
        transform_fn=transform_invoice_data,
    )

    print(result_df)
```

## API Reference

### `extract_structured_data()` Function

```python
def extract_structured_data(
    image_paths: List[str],
    output_cls: Type[BaseModel],
    prompt: str,
    id_column: str = "document_id",
    fields: Optional[List[str]] = None,
    preprocess: bool = False,
    output_dir: Optional[Path] = None,
    scale_factor: int = 3,
    transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame
```

**Required Parameters:**
- `image_paths`: List of document/image paths
- `output_cls`: Pydantic model class for extraction
- `prompt`: Extraction prompt template (must include `{context_str}`)

**Optional Parameters:**
- `id_column`: Document ID column name (default: "document_id")
- `fields`: Fields to extract (default: all model fields)
- `preprocess`: Enable image preprocessing (default: False)
- `output_dir`: Directory for preprocessed images
- `scale_factor`: Image scaling factor (default: 3)
- `transform_fn`: Custom transformation function

**Returns:**
- `pd.DataFrame`: Extracted data

## Usage Examples

### Basic Extraction

```python
from document_extraction_pipeline import extract_structured_data
from pydantic import BaseModel, Field

class BusinessCard(BaseModel):
    name: str = Field(description="Person's name")
    company: str = Field(description="Company name")
    email: str = Field(description="Email address")

result = extract_structured_data(
    image_paths=["card.jpg"],
    output_cls=BusinessCard,
    prompt="Extract business card info: {context_str}",
)
```

### With Image Preprocessing

```python
from pathlib import Path
from extract_receipts_pipeline import Receipt

result = extract_structured_data(
    image_paths=["low_res.jpg"],
    output_cls=Receipt,
    prompt="Extract receipt: {context_str}",
    preprocess=True,
    output_dir=Path("processed_images"),
    scale_factor=3,
)
```

### With Custom Transformations

```python
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["name"] = df["name"].str.title()
    df["email"] = df["email"].str.lower()
    return df

result = extract_structured_data(
    image_paths=["form.pdf"],
    output_cls=FormData,
    prompt="Extract: {context_str}",
    transform_fn=clean_data,
)
```

## Creating New Document Extractors

To create a new document extractor (like the receipt pipeline):

1. Import the generic `extract_structured_data` function from `document_extraction_pipeline`
2. Define your Pydantic schema(s)
3. (Optional) Create transformation function
4. Define extraction prompt
5. Add `__main__` block with example usage

See [extract_receipts_pipeline.py](extract_receipts_pipeline.py) for a complete example.

## Dependencies

Both files include uv inline script dependencies. Required packages:
- llama-index
- llama-index-program-openai
- llama-parse
- python-dotenv
- pandas
- pillow

Run with `uv run <script_name>.py` - dependencies will be automatically installed.
