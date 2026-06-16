"""
NutriVida — Diabetes Risk Prediction API
=========================================
Hierarchical ML model serving:
  Model 1: Diabetes vs No-Diabetes (custom threshold)
  Model 2: Prediabetes vs Normal (among non-diabetics)

NHANES Features:
  RIDAGEYR, RIAGENDR, BMXBMI, BMXWAIST,
  LBXSGL, LBDSTRSI, LBDHDD, BPXSY1, BPXDI1
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Logging ────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrivida-api")

# ─── Model Loading ──────────────────────────────
MODEL_PATH = Path(__file__).parent / "model" / "diabetes_web_model.pkl"

logger.info(f"Loading model from: {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)

model1 = model_bundle["model1"]
model2 = model_bundle["model2"]
threshold_m1 = model_bundle["threshold_m1"]
logger.info(f"Models loaded. Threshold M1: {threshold_m1}")

# ─── Feature order (must match training) ────────
FEATURE_COLS = [
    "RIDAGEYR", "RIAGENDR", "BMXBMI", "BMXWAIST",
    "LBXSGL", "LBDSTRSI", "LBDHDD", "BPXSY1", "BPXDI1",
]

# ─── Pydantic Schemas ───────────────────────────
class HealthInput(BaseModel):
    """
    Input data from the web form.
    The frontend sends raw user values; conversions happen here.
    """
    RIDAGEYR: float = Field(..., ge=1, le=120, description="Edad biológica (años)")
    RIAGENDR: int   = Field(..., ge=1, le=2, description="Sexo biológico (1=Hombre, 2=Mujer)")
    BMXBMI: float   = Field(..., ge=10, le=60, description="Índice de Masa Corporal (kg/m²)")
    BMXWAIST: float = Field(..., ge=40, le=200, description="Perímetro de cintura (cm)")
    LBXSGL: float   = Field(..., ge=30, le=500, description="Glucosa en ayunas (mg/dL)")
    LBDSTRSI: float = Field(..., ge=0.1, le=20, description="Triglicéridos (mmol/L, ya convertido)")
    LBDHDD: float   = Field(..., ge=10, le=150, description="Colesterol HDL (mg/dL)")
    BPXSY1: float   = Field(..., ge=60, le=250, description="Presión arterial sistólica (mmHg)")
    BPXDI1: float   = Field(..., ge=30, le=150, description="Presión arterial diastólica (mmHg)")


class PredictionOutput(BaseModel):
    """Structured prediction response."""
    prediction: str  # "Normal", "Prediabetes", "Diabetes"
    risk_level: str  # "low", "moderate", "high"
    probability_diabetes: float
    probability_prediabetes: float | None = None
    description: str


# ─── FastAPI App ────────────────────────────────
app = FastAPI(
    title="NutriVida Diabetes Prediction API",
    version="1.0.0",
    description="Hierarchical ML prediction for diabetes risk assessment.",
)

# CORS — allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model": "diabetes_web_model", "threshold_m1": threshold_m1}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: HealthInput):
    """
    Hierarchical prediction:
      1. Model 1 checks: Diabetes vs No-Diabetes (custom threshold)
      2. If No-Diabetes → Model 2 checks: Prediabetes vs Normal
    """
    try:
        # Build DataFrame with correct column order
        input_df = pd.DataFrame([{
            col: getattr(data, col) for col in FEATURE_COLS
        }])

        logger.info(f"Input data: {input_df.to_dict(orient='records')[0]}")

        # ── Step 1: Diabetes check ──
        proba_m1 = model1.predict_proba(input_df)[0]
        prob_diabetes = float(proba_m1[1])

        logger.info(f"Model 1 — P(diabetes): {prob_diabetes:.4f}, threshold: {threshold_m1}")

        if prob_diabetes >= threshold_m1:
            return PredictionOutput(
                prediction="Diabetes",
                risk_level="high",
                probability_diabetes=round(prob_diabetes, 4),
                probability_prediabetes=None,
                description=(
                    "El modelo indica una alta probabilidad de diabetes tipo 2. "
                    "Es fundamental que consultes a un médico especialista para una "
                    "evaluación clínica completa y confirmar el diagnóstico."
                ),
            )

        # ── Step 2: Prediabetes check (only for non-diabetics) ──
        proba_m2 = model2.predict_proba(input_df)[0]
        prob_prediabetes = float(proba_m2[1])

        logger.info(f"Model 2 — P(prediabetes): {prob_prediabetes:.4f}")

        if prob_prediabetes >= 0.5:
            return PredictionOutput(
                prediction="Prediabetes",
                risk_level="moderate",
                probability_diabetes=round(prob_diabetes, 4),
                probability_prediabetes=round(prob_prediabetes, 4),
                description=(
                    "Tus indicadores sugieren un estado de prediabetes. "
                    "Esto significa que tus niveles de glucosa son más altos de lo normal, "
                    "pero aún no alcanzan el rango diabético. Consulta a un profesional de "
                    "la salud para ajustar tu dieta y hábitos de vida."
                ),
            )

        return PredictionOutput(
            prediction="Normal",
            risk_level="low",
            probability_diabetes=round(prob_diabetes, 4),
            probability_prediabetes=round(prob_prediabetes, 4),
            description=(
                "Tus indicadores se encuentran dentro de rangos normales. "
                "Mantén un estilo de vida saludable con actividad física regular "
                "y una alimentación equilibrada para conservar estos buenos resultados."
            ),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
