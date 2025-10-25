import google.generativeai as genai
import pandas as pd
import io
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def chat_excel(file_bytes, prompt):
    df = pd.read_excel(io.BytesIO(file_bytes))
    df_sample = df.head(10).to_string()
    full_prompt = f"""
    You are an AI data assistant. Here is a preview of the Excel data:
    {df_sample}
    User question:
    {prompt}
    Answer precisely based only on the given data.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(full_prompt)
    return response.text.strip()
