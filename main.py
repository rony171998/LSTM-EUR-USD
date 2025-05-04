from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.predict import predict_future_prices, PredictRequest
from api.models import router as models_router
import uvicorn

app = FastAPI(
    title="LSTM Prediction API",
    description="API para predecir precios futuros usando un modelo LSTM entrenado",
    version="1.0.0"
)

# Configurar CORS si lo vas a consumir desde el frontend (Next.js por ejemplo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir el router de modelos
app.include_router(models_router, prefix="/api", tags=["models"])

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        result = predict_future_prices(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
