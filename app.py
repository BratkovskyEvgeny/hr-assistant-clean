import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="HR Assistant API")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7


@app.get("/")
def read_root():
    return {"status": "running", "model": os.getenv("MODEL_NAME", "gpt2")}


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        # Здесь будет логика генерации текста через vLLM
        return {
            "generated_text": "Пример сгенерированного текста",
            "model": os.getenv("MODEL_NAME", "gpt2"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
