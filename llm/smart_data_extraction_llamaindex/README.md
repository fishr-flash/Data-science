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
├── document_extraction_pipeline.py        # Generic pipeline (synchronous)
├── async_document_extraction_pipeline.py  # Generic pipeline (async, parallel)
├── extract_receipts_pipeline.py           # Receipt extraction (sync + async functions)
├── async_extract_receipts_pipeline.py     # Receipt extraction (pure async)
└── README.md                               # This file
```

## Performance: Sync vs Async

The pipeline is available in two versions:

- **Synchronous** (`document_extraction_pipeline.py`): Simple, sequential processing
- **Asynchronous** (`async_document_extraction_pipeline.py`): Parallel processing with 3-10x speedup

For batch processing of multiple documents, the async version provides significant performance improvements through concurrent LLM calls and parallel image processing.

## Quick Start

### Option 1: Use the Receipt Pipeline

**Async Version (Recommended for batch processing):**

```bash
uv run async_extract_receipts_pipeline.py
```

**Sync Version (Simple, sequential processing):**

```bash
uv run extract_receipts_pipeline.py
```

Both receipt pipelines include:
- `Receipt` and `ReceiptItem` Pydantic schemas
- Receipt-specific data transformations (uppercase company, parse dates)
- Pre-configured extraction prompt
- Image scaling for better OCR
- Example usage in `__main__` block

The async version processes 4 receipts in parallel by default and includes progress indicators.

### Option 2: Create Your Own Pipeline (Synchronous)

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

### Option 3: Create Your Own Pipeline (Async - Recommended for Batch Processing)

Import the async pipeline for better performance with multiple documents:

```python
import asyncio
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from async_document_extraction_pipeline import extract_structured_data_async


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


# 4. Run extraction with async
async def main():
    invoice_paths = ["invoice1.pdf", "invoice2.pdf", "invoice3.pdf"]

    result_df = await extract_structured_data_async(
        image_paths=invoice_paths,
        output_cls=Invoice,
        prompt=INVOICE_PROMPT,
        id_column="invoice_id",
        data_transformer=transform_invoice_data,
        num_workers=4,  # Process 4 documents in parallel
    )

    print(result_df)


if __name__ == "__main__":
    asyncio.run(main())
```

**Performance Benefits:**
- **Sync version**: Processes documents sequentially (1 at a time)
- **Async version**: Processes documents in parallel (4+ at a time)
- **Expected speedup**: 3-10x faster depending on `num_workers` and API rate limits

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

### `extract_structured_data_async()` Function

```python
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
) -> pd.DataFrame
```

**Additional Parameters:**
- `num_workers`: Number of parallel workers for processing (default: 4). Higher values increase concurrency but may hit API rate limits.

**Performance Characteristics:**
- Uses `asyncio` and `run_jobs()` for parallel processing
- Concurrent LLM extraction calls (limited by `num_workers`)
- Parallel image preprocessing
- Non-blocking I/O operations

**Usage:**
Must be called with `await` inside an async function or using `asyncio.run()`.

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

### Async Batch Processing

```python
import asyncio
from async_document_extraction_pipeline import extract_structured_data_async

async def process_large_batch():
    """Process 100+ documents efficiently with parallel workers."""
    document_paths = [f"doc_{i}.pdf" for i in range(100)]

    result = await extract_structured_data_async(
        image_paths=document_paths,
        output_cls=MySchema,
        prompt="Extract: {context_str}",
        num_workers=6,  # Process 6 documents concurrently
    )

    return result

# Run the async function
result_df = asyncio.run(process_large_batch())
```

**When to use async:**
- Processing 10+ documents
- Time-sensitive batch operations
- High-volume document extraction
- When you need maximum throughput

**When to use sync:**
- Single document extraction
- Simple scripts
- Learning/prototyping
- When simplicity is preferred over performance

### Import and Use Receipt Extractors

**Async receipt extractor:**

```python
import asyncio
from pathlib import Path
from async_extract_receipts_pipeline import extract_receipts

async def process_my_receipts():
    paths = [Path("receipt1.jpg"), Path("receipt2.jpg")]
    result = await extract_receipts(
        receipt_paths=paths,
        output_dir=Path("processed"),
        num_workers=6  # Higher parallelism for more receipts
    )
    return result

# Run it
df = asyncio.run(process_my_receipts())
print(df)
```

**Sync receipt extractor:**

```python
from pathlib import Path
from extract_receipts_pipeline import extract_receipts_sync

paths = [Path("receipt1.jpg"), Path("receipt2.jpg")]
result = extract_receipts_sync(
    receipt_paths=paths,
    output_dir=Path("processed")
)
print(result)
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

**Examples:**
- **Sync**: [extract_receipts_pipeline.py](extract_receipts_pipeline.py) - includes both sync and async functions
- **Async**: [async_extract_receipts_pipeline.py](async_extract_receipts_pipeline.py) - pure async implementation with progress tracking

## Dependencies

### Generic Pipeline (Both Sync and Async)
Required packages (in `document_extraction_pipeline.py` and `async_document_extraction_pipeline.py`):
- llama-index
- llama-index-program-openai
- llama-parse
- python-dotenv
- pandas

### Receipt Pipeline
Additional packages (in `extract_receipts_pipeline.py`):
- pillow (for image preprocessing)

Run with `uv run <script_name>.py` - dependencies will be automatically installed.
