# 🩺 Predictor de Riesgo de Diabetes

Aplicación web de predicción de riesgo de diabetes tipo 2, basada en un modelo jerárquico de Machine Learning entrenado con datos de la encuesta [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm).

> **Repositorio principal:** [Proyecto_Nhanes_Kedro](https://github.com/k3na1/Proyecto_Nhanes_Kedro.git) — rama `DevFfix`
>
> Este repositorio contiene exclusivamente el **frontend** y la **API** de servicio del modelo. El pipeline completo de datos (extracción, transformación, entrenamiento) se encuentra en el repositorio principal de Kedro.

---

## 📁 Estructura del proyecto

```
Web_/
├── API_web/
│   ├── main.py                      # API FastAPI (servidor de predicción)
│   └── model/
│       └── diabetes_web_model.pkl   # Modelo ML jerárquico (XGBoost)
├── web/
│   ├── index.html                   # Página principal
│   ├── styles.css                   # Estilos (diseño editorial, flat)
│   └── app.js                       # Lógica del frontend
└── README.md
```

---

## 🧠 ¿Cómo funciona el modelo?

El sistema utiliza una **clasificación jerárquica** con dos modelos XGBoost:

1. **Modelo 1** — ¿Tiene diabetes? Si la probabilidad ≥ 0.69 → **Diabetes**
2. **Modelo 2** — Si no es diabetes, ¿tiene prediabetes? Si la probabilidad ≥ 0.5 → **Prediabetes**, de lo contrario → **Normal**

### Variables de entrada (NHANES)

| Variable | Descripción | Unidad |
|---|---|---|
| `RIDAGEYR` | Edad biológica | años |
| `RIAGENDR` | Sexo biológico | 1=Hombre, 2=Mujer |
| `BMXBMI` | Índice de Masa Corporal | kg/m² (auto-calculado) |
| `BMXWAIST` | Perímetro de cintura | cm |
| `LBXSGL` | Glucosa en ayunas | mg/dL |
| `LBDSTRSI` | Triglicéridos | mmol/L (convertido desde mg/dL × 0.01129) |
| `LBDHDD` | Colesterol HDL | mg/dL |
| `BPXSY1` | Presión arterial sistólica | mmHg |
| `BPXDI1` | Presión arterial diastólica | mmHg |

---

## 🚀 Inicio rápido

### Requisitos previos

- **Python 3.10+**
- **Node.js** (solo para el servidor de archivos estáticos; opcional si usas otro)
- Paquetes Python:
  ```
  fastapi
  uvicorn
  scikit-learn
  xgboost
  pandas
  numpy
  ```

  Instalación rápida:
  ```bash
  pip install fastapi uvicorn scikit-learn xgboost pandas numpy
  ```

### 1. Iniciar la API (backend)

Abre una terminal y ejecuta:

```bash
cd API_web
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

La API estará disponible en:
- **Servidor:** http://localhost:8000
- **Documentación Swagger:** http://localhost:8000/docs

### 2. Iniciar la página web (frontend)

Abre una **segunda terminal** y ejecuta:

```bash
cd web
npx -y http-server . -p 8080 -c-1
```

La página estará disponible en:
- **Web:** http://localhost:8080

> **Nota:** La opción `-c-1` desactiva el caché del navegador para evitar que se sirvan archivos desactualizados durante el desarrollo.

---

## 🔌 Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Health check — verifica que el modelo esté cargado |
| `POST` | `/predict` | Predicción jerárquica de riesgo de diabetes |

### Ejemplo de petición

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RIDAGEYR": 45,
    "RIAGENDR": 1,
    "BMXBMI": 28.5,
    "BMXWAIST": 95,
    "LBXSGL": 110,
    "LBDSTRSI": 1.69,
    "LBDHDD": 45,
    "BPXSY1": 130,
    "BPXDI1": 85
  }'
```

### Ejemplo de respuesta

```json
{
  "prediction": "Prediabetes",
  "risk_level": "moderate",
  "probability_diabetes": 0.3587,
  "probability_prediabetes": 0.8528,
  "description": "Tus indicadores sugieren un estado de prediabetes..."
}
```

---

## ⚠️ Aviso

Esta herramienta es **orientativa** y no reemplaza el diagnóstico médico profesional. Consulta siempre a un especialista de la salud.
