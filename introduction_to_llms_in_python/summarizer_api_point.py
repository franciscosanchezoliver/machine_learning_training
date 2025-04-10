from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")

app = FastAPI()


class TextToSummarize(BaseModel):
    text: str
    max_length: int = 50


@app.post("/summarize")
def summarize_text(data: TextToSummarize):
    """
    Summarizes the given text using a pre-trained model.

    Parameters:
    - data: TextToSummarize object containing the text to summarize and max_length.

    Returns:
    - A dictionary containing the summary.
    """
    try:
        summary = summarizer(data.text, max_length=data.max_length)
        return {"summary": summary[0]["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
