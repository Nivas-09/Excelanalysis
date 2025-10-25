import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from fastapi import UploadFile
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Import preprocessing function
from service.preprocesss import preprocess_excel

async def analyze_excel(file: UploadFile):
    """
    Analyze Excel file: Preprocess + Generate charts + AI Summary + Recommendations
    (Removed premium Imagen image generation feature)
    """
    try:
        # Step 1: First preprocess the data
        preprocess_result = await preprocess_excel(file)
        
        if not preprocess_result.get("success"):
            return preprocess_result
        
        # Get the preprocessed file path
        cleaned_file_path = preprocess_result["data"]["cleaned_file_path"]
        df = pd.read_excel(cleaned_file_path)
        
        print(f"DEBUG: Starting analysis for dataset shape: {df.shape}")
        
        # Step 2: Generate AI Summary using Gemini API
        ai_summary = await generate_ai_summary(df)
        print("DEBUG: AI Summary generated")
        
        # Step 3: Generate basic visualizations
        charts = await generate_visualizations(df)
        print(f"DEBUG: Charts generated: {type(charts)}, keys: {list(charts.keys()) if isinstance(charts, dict) else 'Not a dict'}")
        
        if not isinstance(charts, dict) or "error" in charts:
            return {
                "success": False,
                "error": "Chart generation failed",
                "message": charts.get("error", "Unknown chart error") if isinstance(charts, dict) else str(charts)
            }
        
        # Step 4: Enhance first 2 charts with AI descriptions
        enhanced_charts = await enhance_charts_with_ai(charts, df)
        print("DEBUG: Enhanced charts generated")
        
        # Step 5: Generate overall recommendations
        overall_recommendations = await generate_overall_recommendations(ai_summary, enhanced_charts, df)
        print("DEBUG: Overall recommendations generated")
        
        # Step 6: Prepare final response
        return {
            "success": True,
            "message": "Analysis completed successfully ✅",
            "data": {
                **preprocess_result["data"],
                "ai_summary": ai_summary,
                "charts": enhanced_charts,
                "overall_recommendations": overall_recommendations,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        print(f"ERROR in analyze_excel: {str(e)}")
        return {
            "success": False,
            "error": "Analysis failed",
            "message": str(e)
        }

async def generate_overall_recommendations(ai_summary: str, enhanced_charts: dict, df: pd.DataFrame) -> str:
    """Generate overall recommendations for the entire analysis"""
    try:
        if not enhanced_charts or not isinstance(enhanced_charts, dict):
            return "No chart insights available for recommendations"
        
        chart_insights = []
        for chart_name, chart_info in list(enhanced_charts.items())[:2]:
            if chart_info and isinstance(chart_info, dict):
                insight = chart_info.get('ai_insight', 'No insight available')
                if "provide the image" in insight.lower() or "send the chart" in insight.lower():
                    insight = f"Chart {chart_name} shows key patterns in the data revealing business insights about {list(df.columns)}."
                chart_insights.append(f"{chart_name}: {insight}")
        
        prompt = f"""
        Based on the data analysis, provide comprehensive recommendations:

        DATASET SUMMARY:
        {ai_summary}

        KEY CHART INSIGHTS:
        {chr(10).join(chart_insights) if chart_insights else 'Key patterns identified in visualizations'}

        DATASET DETAILS:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Numeric columns: {list(df.select_dtypes(include=[np.number]).columns)}
        - Categorical columns: {list(df.select_dtypes(include=['object']).columns)}

        Include:
        1. Data strategy improvements
        2. Business actions
        3. Analysis roadmap
        4. Visualization presentation tips
        5. Long-term sustainability
        """

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"ERROR in generate_overall_recommendations: {str(e)}")
        return f"Overall recommendations generation failed: {str(e)}"

async def generate_ai_summary(df: pd.DataFrame) -> str:
    """Dataset AI Summary"""
    try:
        dataset_info = f"""
        Dataset Shape: {df.shape}
        Columns: {list(df.columns)}
        Data Types: {df.dtypes.to_dict()}
        Sample Data (first 5 rows):
        {df.head().to_string()}
        Basic Statistics:
        {df.describe().to_string() if not df.select_dtypes(include=[np.number]).empty else 'No numeric columns'}
        Missing Values: {df.isnull().sum().to_dict()}
        """
        
        prompt = f"""
        Analyze this dataset and summarize:
        {dataset_info}
        Include:
        1. Dataset overview
        2. Key insights and patterns
        3. Data quality observations
        4. Analysis opportunities
        5. Limitations or improvements
        Keep it concise and business-focused.
        """

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"AI summary generation failed: {str(e)}"

async def enhance_charts_with_ai(charts: dict, df: pd.DataFrame) -> dict:
    """Add AI insights to charts (text-only, no image generation)"""
    try:
        if not charts or not isinstance(charts, dict):
            return {"error": "No charts available for AI enhancement"}
        
        enhanced_charts = {}
        charts_to_process = dict(list(charts.items())[:2])
        print(f"DEBUG: Applying AI insights to charts: {list(charts_to_process.keys())}")

        for chart_name, chart_data in charts_to_process.items():
            if chart_data and not chart_data.startswith("Error"):
                chart_details = await get_chart_details(chart_name, df)
                
                prompt = f"""
                Analyze this chart and provide insights:

                CHART DETAILS:
                {chart_details}

                DATASET CONTEXT:
                Shape: {df.shape}
                Columns: {list(df.columns)}

                Provide:
                1. What the chart shows
                2. Key business insights
                3. Trends or anomalies
                4. Actionable recommendations
                """

                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                
                enhanced_charts[chart_name] = {
                    "chart_image": chart_data,
                    "ai_insight": response.text,
                    "has_ai_analysis": True
                }
        
        # Keep remaining charts basic
        for chart_name, chart_data in list(charts.items())[2:]:
            enhanced_charts[chart_name] = {
                "chart_image": chart_data,
                "ai_insight": "Basic chart - no AI analysis applied",
                "has_ai_analysis": False
            }
        
        return enhanced_charts
        
    except Exception as e:
        print(f"ERROR in enhance_charts_with_ai: {str(e)}")
        return {"error": f"AI enhancement failed: {str(e)}"}

async def get_chart_details(chart_name: str, df: pd.DataFrame) -> str:
    """Extract details about chart for AI context"""
    try:
        if 'barchart' in chart_name.lower():
            col_name = chart_name.replace('barchart_', '')
            if col_name in df.columns:
                value_counts = df[col_name].value_counts().head(10)
                return f"Bar Chart for '{col_name}' - top 10 values: {dict(value_counts)}"
        elif 'piechart' in chart_name.lower():
            col_name = chart_name.replace('piechart_', '')
            if col_name in df.columns:
                value_counts = df[col_name].value_counts().head(8)
                return f"Pie Chart for '{col_name}' - value distribution: {dict(value_counts)}"
        elif 'correlation' in chart_name.lower():
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            return f"Correlation Heatmap for numeric columns: {numeric_cols}"
        
        return f"Chart {chart_name} showing dataset columns: {list(df.columns)}"
        
    except Exception:
        return f"Chart {chart_name} showing dataset columns: {list(df.columns)}"

async def generate_visualizations(df: pd.DataFrame) -> dict:
    """Generate basic matplotlib/seaborn charts"""
    charts = {}
    try:
        plt.style.use('default')
        sns.set_palette("husl")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"DEBUG: Numeric columns: {numeric_cols}")
        print(f"DEBUG: Categorical columns: {categorical_cols}")

        # BAR CHARTS for first few categorical columns
        if len(categorical_cols) > 0:
            for i, col in enumerate(categorical_cols[:3]):
                plt.figure(figsize=(12, 6))
                value_counts = df[col].value_counts().head(10)
                if len(value_counts) > 0:
                    bars = plt.bar(range(len(value_counts)), value_counts.values, 
                                  color='lightcoral', alpha=0.7, edgecolor='black')
                    plt.title(f'Top 10 Values in {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                    for bar, value in zip(bars, value_counts.values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                 str(value), ha='center', va='bottom')
                    plt.tight_layout()
                    charts[f'barchart_{col}'] = save_plot_as_base64()
        
        print(f"DEBUG: Generated {len(charts)} charts")
        return charts
        
    except Exception as e:
        print(f"ERROR in generate_visualizations: {str(e)}")
        return {"error": f"Chart generation failed: {str(e)}"}

def save_plot_as_base64() -> str:
    """Convert Matplotlib plot to base64 image"""
    try:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        return f"Error generating chart: {str(e)}"
