import pandas as pd
import numpy as np
import os
import io
from datetime import datetime
from fastapi import UploadFile

async def preprocess_excel(file: UploadFile):
    """
    Preprocess Excel file and calculate data quality score
    Returns cleaned Excel file and data score
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        # Check if DataFrame is empty
        if df.empty:
            return {
                "success": False,
                "error": "Empty file",
                "message": "Uploaded Excel file is empty"
            }

        # Store original data for score calculation
        df_original = df.copy()
        original_rows, original_cols = df_original.shape

        # ========== PREPROCESSING STEPS ==========

        # 1️⃣ Remove columns with all null values
        cols_before = len(df.columns)
        df = df.dropna(axis=1, how='all')
        null_columns_removed = cols_before - len(df.columns)

        # 2️⃣ Remove duplicate rows
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = duplicates_before

        # 3️⃣ Handle missing values
        missing_before = df_original.isnull().sum().sum()
        
        # Smart imputation by data type
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        if not numeric_cols.empty:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical columns with mode
        if not categorical_cols.empty:
            for col in categorical_cols:
                if not df[col].empty:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
        
        # Final fill any remaining nulls
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        missing_after = df.isnull().sum().sum()

        # 4️⃣ Normalize column names
        df.columns = [str(col).strip().replace(" ", "_").lower() for col in df.columns]

        # ========== DATA QUALITY SCORE CALCULATION ==========
        data_score = calculate_data_score(
            original_rows=original_rows,
            original_cols=original_cols,
            missing_before=missing_before,
            missing_after=missing_after,
            duplicates_removed=duplicates_removed
        )

        # ========== SAVE PROCESSED FILE ==========
        os.makedirs("processed_data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{file.filename}"
        output_path = os.path.join("processed_data", output_filename)
        
        df.to_excel(output_path, index=False)

        # ========== PREPARE RESPONSE ==========
        return {
            "success": True,
            "message": "Preprocessing completed successfully ✅",
            "data": {
                "original_rows": original_rows,
                "original_columns": original_cols,
                "processed_rows": df.shape[0],
                "processed_columns": df.shape[1],
                "missing_values_before": int(missing_before),
                "missing_values_after": int(missing_after),
                "duplicates_removed": int(duplicates_removed),
                "null_columns_removed": null_columns_removed,
                "data_quality_score": data_score,
                "cleaned_file_path": output_path,
                "file_download_url": f"/download/{output_filename}"  # For frontend
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": "Processing error",
            "message": str(e)
        }

def calculate_data_score(original_rows, original_cols, missing_before, missing_after, duplicates_removed):
    """
    Calculate data quality score (0-100)
    """
    total_cells = original_rows * original_cols
    
    # Safety check
    if total_cells == 0:
        return 0
    
    # 1. Completeness Score (40%) - How complete is data after cleaning
    completeness = max(0, 100 - (missing_after / total_cells * 100))
    
    # 2. Improvement Score (30%) - How much we improved missing values
    if missing_before > 0:
        improvement = ((missing_before - missing_after) / missing_before * 100)
    else:
        improvement = 100  # Perfect if no missing values initially
    improvement = max(0, min(100, improvement))
    
    # 3. Consistency Score (30%) - Duplicate handling
    if original_rows > 0:
        consistency = max(0, 100 - (duplicates_removed / original_rows * 100))
    else:
        consistency = 100
        
    # Weighted final score
    final_score = (0.4 * completeness) + (0.3 * improvement) + (0.3 * consistency)
    return max(0, min(100, round(final_score, 2)))