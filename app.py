"""Локальное приложение для определения тональности отзыва."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from config import MODEL_DIR
from inference import load_fine_tuned, predict_fine_tuned


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Введите непустой отзыв")
        return value


@asynccontextmanager
async def lifespan(app):
    if not (MODEL_DIR / "model.safetensors").exists():
        raise RuntimeError(
            "Модель не найдена. Выполните обучение или загрузите готовые модели по README"
        )
    app.state.model, app.state.tokenizer = load_fine_tuned()
    app.state.lock = threading.Lock()
    yield


app = FastAPI(title="Transformer Sentiment", lifespan=lifespan)


@app.post("/predict")
def predict(request: PredictionRequest):
    with app.state.lock:
        return predict_fine_tuned(request.text, app.state.model, app.state.tokenizer)[0]


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transformer Sentiment</title>
<style>body{max-width:680px;margin:60px auto;padding:0 24px;font:18px system-ui;color:#20222a;background:#f7f6f2}textarea{box-sizing:border-box;width:100%;padding:15px;font:inherit;border:1px solid #aaa;border-radius:8px}button{margin-top:16px;padding:12px 22px;background:#283f35;color:white;border:0;border-radius:8px;font:inherit}pre{white-space:pre-wrap;line-height:1.6}small{color:#555}</style>
<h1>Тональность отзыва</h1><p>Введите отзыв о фильме на английском. Модель выбирает один из двух классов: Negative или Positive.</p><small>Анализируются первые 128 токенов. Вероятности показывают оценки модели.</small>
<form id="form"><textarea id="text" rows="6" required maxlength="20000" placeholder="This movie was absolutely fantastic!"></textarea><button id="button">Определить тональность</button></form><pre id="result" role="status"></pre>
<script>document.querySelector('#form').onsubmit=async e=>{e.preventDefault();const b=document.querySelector('#button'),o=document.querySelector('#result');b.disabled=true;o.textContent='Анализируем…';try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.querySelector('#text').value})});const d=await r.json();if(!r.ok)throw Error('Проверьте текст и повторите запрос.');o.textContent=d.label+'\\n'+Object.entries(d.probabilities).map(([k,v])=>k+': '+(v*100).toFixed(1)+'%').join('\\n')}catch(e){o.textContent=e.message}finally{b.disabled=false}};</script></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
