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


# 2. Optional: Define data transformer
def transform_invoice_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform extracted invoice data."""
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
        data_transformer=transform_invoice_data,
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
    image_transform_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
    image_output_dir: Optional[Path] = None,
    data_transformer: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame
```

**Required Parameters:**
- `image_paths`: List of document/image paths
- `output_cls`: Pydantic model class for extraction
- `prompt`: Extraction prompt template (must include `{context_str}`)

**Optional Parameters:**
- `id_column`: Document ID column name (default: "document_id")
- `fields`: Fields to extract (default: all model fields)
- `image_transform_fn`: Optional function to transform images (takes PIL Image, returns PIL Image)
- `image_output_dir`: Directory to save transformed images (required if image_transform_fn provided)
- `data_transformer`: Optional function to transform the extracted DataFrame

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

### With Image Transformation

```python
from pathlib import Path
from PIL import Image
from extract_receipts_pipeline import Receipt, scale_receipt_image

result = extract_structured_data(
    image_paths=["low_res.jpg"],
    output_cls=Receipt,
    prompt="Extract receipt: {context_str}",
    image_transform_fn=scale_receipt_image,  # Simple function reference
    image_output_dir=Path("processed_images"),
)
```

### With Data Transformation

```python
import pandas as pd

def transform_form_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize extracted form data."""
    df = df.copy()
    df["name"] = df["name"].str.title()
    df["email"] = df["email"].str.lower()
    return df

result = extract_structured_data(
    image_paths=["form.pdf"],
    output_cls=FormData,
    prompt="Extract: {context_str}",
    data_transformer=transform_form_data,
)
```

## Creating New Document Extractors

To create a new document extractor (like the receipt pipeline):

1. Import the generic `extract_structured_data` function from `document_extraction_pipeline`
2. Define your Pydantic schema(s)
3. (Optional) Create `image_transform_fn` - a simple function that transforms one PIL Image
4. (Optional) Create `data_transformer` function for data transformation
5. Define extraction prompt
6. Add `__main__` block with example usage

**Example image transformation:**
```python
from PIL import Image

def rotate_and_scale(img: Image.Image) -> Image.Image:
    """Custom transformation: rotate 90 degrees and scale up."""
    rotated = img.rotate(90, expand=True)
    new_size = (rotated.width * 2, rotated.height * 2)
    return rotated.resize(new_size, Image.Resampling.LANCZOS)
```

See [extract_receipts_pipeline.py](extract_receipts_pipeline.py) for a complete example.

## Dependencies

### Generic Pipeline
Required packages (in `document_extraction_pipeline.py`):
- llama-index
- llama-index-program-openai
- llama-parse
- python-dotenv
- pandas

### Receipt Pipeline
Additional packages (in `extract_receipts_pipeline.py`):
- pillow (for image preprocessing)

Run with `uv run <script_name>.py` - dependencies will be automatically installed.
