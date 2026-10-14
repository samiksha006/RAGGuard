"""FastAPI service for the trained TF-IDF baseline; input validation and health checks included."""
import os, time
import json
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app=FastAPI(title='RAGGuard API',version='1.0.0',description='RAG answer hallucination-risk estimation. Scores are not proof of factuality.')
MODEL_PATH=Path(os.getenv('RAGGUARD_MODEL_PATH','artifacts/text_baseline/model.joblib'))
THRESHOLD_PATH=Path(os.getenv('RAGGUARD_THRESHOLD_PATH','artifacts/text_baseline/selected_threshold.json'))
_model=None
class PredictRequest(BaseModel):
    question: str = Field(...,min_length=1,max_length=5000)
    context: str = Field(...,min_length=1,max_length=20000)
    answer: str = Field(...,min_length=1,max_length=10000)
    @field_validator('question','context','answer')
    @classmethod
    def nonblank(cls,v):
        if not v.strip(): raise ValueError('Field must not be blank')
        return v.strip()
class PredictResponse(BaseModel):
    prediction: str
    hallucination_risk: float
    threshold: float
    model: str
    latency_ms: float
    disclaimer: str

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists(): raise HTTPException(status_code=503,detail=f'Model not found at {MODEL_PATH}. Train the baseline first or mount the artifacts directory.')
        _model=joblib.load(MODEL_PATH)
    return _model
def get_threshold():
    configured=os.getenv('RAGGUARD_THRESHOLD')
    if configured is not None:
        try: threshold=float(configured)
        except ValueError as e: raise HTTPException(status_code=500,detail='RAGGUARD_THRESHOLD must be a number between 0 and 1.') from e
    elif THRESHOLD_PATH.exists():
        data=json.loads(THRESHOLD_PATH.read_text(encoding='utf-8'))
        try: threshold=float(data['threshold'])
        except (KeyError,TypeError,ValueError) as e: raise HTTPException(status_code=500,detail=f'Invalid threshold file: {THRESHOLD_PATH}') from e
    else:
        threshold=0.5
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=500,detail='The configured decision threshold must be between 0 and 1.')
    return threshold
@app.get('/health')
def health(): return {'status':'ok','model_available':MODEL_PATH.exists(),'model_path':str(MODEL_PATH)}
@app.get('/')
def root(): return {'service':'RAGGuard','docs':'/docs','health':'/health'}
@app.post('/predict',response_model=PredictResponse)
def predict(req:PredictRequest):
    model=get_model(); text=f'{req.question} [CONTEXT] {req.context} [ANSWER] {req.answer}'; start=time.perf_counter()
    try: risk=float(model.predict_proba([text])[0,1])
    except Exception as e: raise HTTPException(status_code=500,detail=f'Inference failed: {type(e).__name__}')
    latency=(time.perf_counter()-start)*1000; threshold=get_threshold()
    return PredictResponse(prediction='hallucination-risk' if risk>=threshold else 'lower-risk',hallucination_risk=round(risk,6),threshold=threshold,model='TF-IDF + Logistic Regression baseline',latency_ms=round(latency,3),disclaimer='This is a statistical risk estimate, not a factuality guarantee. Verify important claims against source evidence.')
