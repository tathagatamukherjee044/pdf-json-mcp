from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import os
from pydantic import BaseModel
from typing import Dict, Any
from config_generation import PDFSimilarityProcessor

app = FastAPI()

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

class ProcessPDFResponse(BaseModel):
    message: str
    config: Dict[str, Any]

@app.post("/process-pdf", response_model=ProcessPDFResponse)
async def process_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    temp_file_path = None
    try:
        # Save the uploaded file temporarily
        temp_file_path = Path("data") / file.filename
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF and get the config
        processor = PDFSimilarityProcessor()
        config_data = processor.process_pdf_and_return_config(str(temp_file_path))
        
        # Clean up the temporary file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        
        return ProcessPDFResponse(
            message="PDF processed successfully",
            config=config_data
        )
        
    except Exception as e:
        # Clean up in case of error
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 