from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os

from fastapi.responses import FileResponse

# Import our service functions (we'll create these files next)
from service import preprocesss
from service import analyses

app = FastAPI(title="Excel Preprocessing & Analysis API")

# Create necessary directories
os.makedirs("temp_files", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)
os.makedirs("analysis_reports", exist_ok=True)

@app.post("/preprocess")
async def preprocess_file(file: UploadFile = File(...)):
    """
    Preprocess Excel file: Clean data, handle missing values, remove duplicates
    Returns: Cleaned Excel file + data quality score
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid file type",
                    "message": "File must be an Excel file (.xlsx, .xls)"
                }
            )
        
        # Call preprocessing service
        result = await preprocesss.preprocess_excel(file)
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Preprocessing failed",
                "message": str(e)
            }
        )

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze Excel file: Preprocess + Generate charts + Text summary
    Returns: Cleaned Excel + Visualizations + Analysis report
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid file type",
                    "message": "File must be an Excel file (.xlsx, .xls)"
                }
            )
        
        # Call analysis service
        result = await analyses.analyze_excel(file)
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Analysis failed",
                "message": str(e)
            }
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download processed Excel files temporarily
    """
    try:
        # Security check
        if ".." in filename or "/" in filename:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid filename"}
            )
        
        file_path = os.path.join("processed_data", filename)
        
        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": "File not found"}
            )
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Download failed", "message": str(e)}
        )