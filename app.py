from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import os
from config_generation import PDFSimilarityProcessor

app = FastAPI()

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

@app.post("/process-pdf")
async def process_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save the uploaded file temporarily
        temp_file_path = Path("data") / file.filename
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF
        processor = PDFSimilarityProcessor()
        processor.process_pdf(str(temp_file_path), "output")
        
        # Get the output file path
        output_file = Path("output") / f"{temp_file_path.stem}_extraction_config.json"
        
        # Clean up the temporary file
        temp_file_path.unlink()
        
        return JSONResponse(
            content={"message": "PDF processed successfully", "output_file": str(output_file)},
            status_code=200
        )
        
    except Exception as e:
        # Clean up in case of error
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 