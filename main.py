import base64
import io
import json
import os
from PIL import Image
import anthropic
import pydantic
import sys
import traceback # Added for more detailed error logging

# LangChain specific imports
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

# For PDF processing (ensure PyMuPDF is installed: pip install PyMuPDF)
import fitz
from io import BytesIO

import os
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic Version Check (UNCHANGED) ---
if pydantic.VERSION.startswith('1.'):
    print("---------------------------------------------------------------------")
    print("WARNING: You are using Pydantic v1. For best compatibility with")
    print("         recent LangChain versions and 'model_json_schema' usage,")
    print("         it is highly recommended to upgrade to Pydantic v2 or later.")
    print("         Run: pip install --upgrade pydantic")
    print("---------------------------------------------------------------------")
elif not pydantic.VERSION.startswith('2.'):
    print(f"WARNING: Unexpected Pydantic version: {pydantic.VERSION}. Expected v1 or v2.")
    print("         Please ensure Pydantic is properly installed.")
    sys.exit(1)

# --- Pydantic Models for the Template JSON (UNCHANGED) ---
class TemplateConfig(BaseModel):
    delimiter: str = Field(description="Delimiter used between elements, typically a space ' '")
    stop_words: str = Field(alias="stopWords", description="Placeholder for stop words, often '_________'")
    invoice_line: int = Field(alias="invoiceLine", description="Number of lines to skip before the invoice table begins (starts from 1)")
    column_length: int = Field(alias="columnLength", description="Number of columns in the main invoice table")

class RemittanceDocConfigItem(BaseModel):
    token: str = Field(description="The specific text token found on the invoice (e.g., 'Pay Number', 'Date')")
    delimiter: str = Field(description="Delimiter for this token's value, typically a space ' '")
    position: int = Field(description="Position of the token's value relative to the token (0-indexed, -1 for last column)")
    format: Optional[str] = Field(None, description="Date format if the token is a date (e.g., 'dd/MM/yyyy')")
    std_nm: str = Field(alias="stdNm", description="The standard name for this token (e.g., 'PABankUtr', 'payDate', 'docId', 'docAmt')")

class InvoiceTemplate(BaseModel):
    template_config: TemplateConfig = Field(alias="templateConfig", description="Configuration for general template properties")
    template_nm: str = Field(alias="templateNm", description="Name of the template (e.g., 'CUSTOM#2_PDF'), not required for extraction logic itself")
    footer: List[str] = Field(description="List of text cues indicating the end of the invoice table")
    header: List[str] = Field(description="List of text cues indicating the start of the invoice table")
    remittance_config: List[RemittanceDocConfigItem] = Field(alias="remittanceConfig", description="Configuration for payment data found outside the main table")
    doc_config: List[RemittanceDocConfigItem] = Field(alias="docConfig", description="Configuration for invoice data found within the main table")

# --- Constants for file paths (UNCHANGED) ---
JSON_TEMPLATE_FILE = "actual.json"
SAMPLE_PDF_FILE = "MULTI.pdf" # IMPORTANT: Replace with your actual PDF filename
TEST_PDF_FILE = "TEST.pdf"

# --- Helper function for PDF to image conversion (MODIFIED FOR ROBUSTNESS) ---
def pdf_page_to_image_base64(pdf_bytes: bytes, page_num: int = 0) -> str:
    """
    Converts a specific page of a PDF (bytes) to a Base64 encoded PNG image.
    Uses PIL explicitly for more reliable image format conversion.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_num >= doc.page_count:
            raise ValueError(f"Page number {page_num} is out of bounds for PDF with {doc.page_count} pages.")
        page = doc.load_page(page_num)
        
        # MODIFICATION HERE: Use 1x resolution (original size)
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1)) # Changed from 2,2 to 1,1

        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        encoded_string = base64.b64encode(img_bytes).decode("utf-8")
        return encoded_string
    except Exception as e:
        print(f"ERROR in pdf_page_to_image_base64: {e}")
        traceback.print_exc()
        raise

# --- LangChain Chain Definition for Template Generation (MODIFIED) ---
def get_template_generation_chain(invoice_image_base64: str):
    llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)

    # Get the data directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    # Initialize messages list for the prompt
    messages = [
        ("system", "You are an AI assistant specialized in creating invoice extraction templates. "
                   "Your task is to analyze a given invoice PDF (or its text/image representation) "
                   "and generate a JSON configuration template that describes how to extract specific data fields. "
                   "Focus on identifying patterns, delimiters, positions, and structural cues like headers and footers for the main invoice table. "
                   "The output must strictly conform to the provided JSON schema for the 'InvoiceTemplate'. "
                   "If you cannot determine a value, use appropriate defaults or omit optional fields. "
                   "\n{format_instructions}")
    ]

    # Load all training examples from the data directory
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.json'):
            # Get the corresponding PDF file (case-insensitive)
            pdf_filename = filename.replace('.json', '.pdf')
            pdf_path = os.path.join(data_dir, pdf_filename)
            
            # Find the actual PDF file (case-insensitive)
            actual_pdf = None
            for f in os.listdir(data_dir):
                if f.lower() == pdf_filename.lower():
                    actual_pdf = f
                    break
            
            if actual_pdf is None:
                print(f"Warning: No matching PDF found for {filename}")
                continue

            # Load JSON template
            json_path = os.path.join(data_dir, filename)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    template_json = json.load(f)
                template_json_string = json.dumps(template_json, indent=2)
            except Exception as e:
                print(f"Warning: Error loading {filename}: {e}")
                continue

            # Load and convert PDF to image
            try:
                with open(os.path.join(data_dir, actual_pdf), 'rb') as f:
                    pdf_bytes = f.read()
                sample_image_b64 = pdf_page_to_image_base64(pdf_bytes)
                
                # Add training example to messages
                messages.extend([
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Here is a training example. First, the invoice image:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sample_image_b64}"}},
                            {"type": "text", "text": "And here is its corresponding template JSON:"},
                            {"type": "text", "text": f"```json\n{template_json_string}\n```"}
                        ]
                    )
                ])
            except Exception as e:
                print(f"Warning: Error processing {actual_pdf}: {e}")
                continue

    # Add the final user query messages
    messages.extend([
        HumanMessage(
            content=[
                {"type": "text", "text": "Now, analyze the following invoice image and generate its corresponding extraction template JSON. Ensure your output is a JSON string conforming to the structure demonstrated above and the Pydantic schema:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{invoice_image_base64}"}}
            ]
        )
    ])

    prompt = ChatPromptTemplate.from_messages(messages)
    parser = PydanticOutputParser(pydantic_object=InvoiceTemplate)
    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser
    return chain

# --- Main execution block (MODIFIED FOR DEBUGGING) ---
async def main():
    print("--- Starting Invoice Template Generation Test ---")

    pdf_bytes = None
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEST_PDF_FILE)
    try:
        print(f"Attempting to load PDF from: {pdf_path}")
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        print(f"Successfully loaded {len(pdf_bytes)} bytes from '{TEST_PDF_FILE}'.")
    except FileNotFoundError:
        print(f"ERROR: Test PDF '{TEST_PDF_FILE}' not found at '{pdf_path}'. "
              "Please ensure your test PDF is in the same directory as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading '{TEST_PDF_FILE}': {e}")
        traceback.print_exc()
        sys.exit(1)

    # Convert the first page of the PDF to a Base64 image


    invoice_image_b64 = ""
    try:
        print("Attempting to convert PDF to Base64 image...")
        invoice_image_b64 = pdf_page_to_image_base64(pdf_bytes)
        print(f"Converted PDF to image. Base64 string length: {len(invoice_image_b64)}")
        print(f"Base64 prefix (first 50 chars): {invoice_image_b64[:50]}...")

        # --- IMPORTANT DEBUGGING STEP ---
        # Try to decode the Base64 string locally and verify it's a valid image.
        # If this fails, the problem is definitely in your conversion.
        try:
            decoded_img_bytes = base64.b64decode(invoice_image_b64)
            # Try to open the image to verify integrity
            with Image.open(io.BytesIO(decoded_img_bytes)) as img:
                img.verify() # Checks image integrity without loading all pixels
                print(f"Local Base64 decode and image verification successful. Image format: {img.format}, size: {img.size}")
        except Exception as e:
            print(f"Local Base64 decode/image verification FAILED! This is the source of the 'invalid base64 data' error.")
            print(f"Error during local verification: {e}")
            traceback.print_exc()
            sys.exit(1) # Exit if local verification fails, no point sending to Claude

    except Exception as e:
        print(f"ERROR: Failed to convert PDF to image: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Get the LangChain processing chain
    template_generation_chain = get_template_generation_chain(invoice_image_b64)

    print("Invoking Claude via LangChain to generate the template...")
    print("claude api key: ", os.getenv("ANTHROPIC_API_KEY"))
    try:
        result: InvoiceTemplate = await template_generation_chain.ainvoke({})

        print("\n--- Generated Invoice Template JSON ---")
        print(result.model_dump_json(indent=2, by_alias=True))
        print("\n--- Test Complete ---")

    except Exception as e:
        import traceback
        print(f"\nERROR: Failed to generate template.")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {e}")
        print("\nPlease check your Claude API key, network connection, and prompt for issues.")
        print("Also, ensure Pydantic is v2.x.x, or if v1.x.x, consider upgrading LangChain or Pydantic.")
        # traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())