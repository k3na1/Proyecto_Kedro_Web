/* ============================================
   NUTRIVIDA — Health Predictor
   Application Logic (Vanilla JS, Modular)
   Connects to FastAPI backend for hierarchical
   diabetes risk prediction.
   ============================================ */

// ─── API Configuration ─────────────────────────
// Dynamically resolve API URL based on current host.
// In production, change the port or path to match your deployment.
const API_HOST = window.location.hostname || "localhost";
const API_PORT = 8000;
const API_URL = `http://${API_HOST}:${API_PORT}/predict`;

// ─── DOM References ────────────────────────────
const DOM = {
  form: document.getElementById("predictorForm"),
  submitBtn: document.getElementById("submitBtn"),
  resetBtn: document.getElementById("resetBtn"),
  // Result states
  resultsIdle: document.getElementById("resultsIdle"),
  resultsLoading: document.getElementById("resultsLoading"),
  resultsCard: document.getElementById("resultsCard"),
  // Result display elements
  riskCircle: document.getElementById("riskCircle"),
  riskValue: document.getElementById("riskValue"),
  riskLevel: document.getElementById("riskLevel"),
  riskDescription: document.getElementById("riskDescription"),
  metricBmi: document.getElementById("metricBmi"),
  metricGlucose: document.getElementById("metricGlucose"),
  metricTriglycerides: document.getElementById("metricTriglycerides"),
  metricHdl: document.getElementById("metricHdl"),
  metricSystolic: document.getElementById("metricSystolic"),
  metricDiastolic: document.getElementById("metricDiastolic"),
  resultTimestamp: document.getElementById("resultTimestamp"),
};

// ─── Form Data Collection ──────────────────────
/**
 * Reads all form inputs, computes BMI internally,
 * and returns the structured payload for the API.
 *
 * @returns {Object} API-ready payload with NHANES variable names
 */
function collectFormData() {
  const height = parseFloat(document.getElementById("fieldHeight").value) || 0;
  const weight = parseFloat(document.getElementById("fieldWeight").value) || 0;
  const heightM = height / 100;
  const bmi = heightM > 0 ? weight / (heightM * heightM) : 0;

  const triglyceridesRaw = parseFloat(document.getElementById("fieldTriglycerides").value) || 0;

  return {
    // Raw values for display in results
    _raw: {
      height,
      weight,
      bmi: parseFloat(bmi.toFixed(1)),
      triglyceridesRaw,
    },
    // API payload — NHANES variable names
    RIDAGEYR: parseFloat(document.getElementById("fieldAge").value) || 0,
    RIAGENDR: parseInt(document.getElementById("fieldSex").value) || 0,
    BMXBMI: parseFloat(bmi.toFixed(2)),
    BMXWAIST: parseFloat(document.getElementById("fieldWaist").value) || 0,
    LBXSGL: parseFloat(document.getElementById("fieldGlucose").value) || 0,
    LBDSTRSI: parseFloat((triglyceridesRaw * 0.01129).toFixed(4)),  // mg/dL → mmol/L
    LBDHDD: parseFloat(document.getElementById("fieldHdl").value) || 0,
    BPXSY1: parseFloat(document.getElementById("fieldSystolic").value) || 0,
    BPXDI1: parseFloat(document.getElementById("fieldDiastolic").value) || 0,
  };
}

/**
 * Validates form data before sending to API.
 * @param {Object} data
 * @returns {{ valid: boolean, errors: string[] }}
 */
function validateFormData(data) {
  const errors = [];

  // Section 1: Datos Personales
  if (!data.RIDAGEYR || data.RIDAGEYR < 1 || data.RIDAGEYR > 120) {
    errors.push("Por favor, introduce una edad válida (1-120 años).");
  }
  if (!data.RIAGENDR || (data.RIAGENDR !== 1 && data.RIAGENDR !== 2)) {
    errors.push("Selecciona tu sexo biológico.");
  }

  // Section 2: Medidas Físicas
  if (!data._raw.height || data._raw.height < 50 || data._raw.height > 250) {
    errors.push("La estatura debe estar entre 50 y 250 cm.");
  }
  if (!data._raw.weight || data._raw.weight < 20 || data._raw.weight > 300) {
    errors.push("El peso debe estar entre 20 y 300 kg.");
  }
  if (!data.BMXWAIST || data.BMXWAIST < 40 || data.BMXWAIST > 200) {
    errors.push("El perímetro de cintura debe estar entre 40 y 200 cm.");
  }

  // Section 3: Laboratorio
  if (!data.LBXSGL || data.LBXSGL < 30 || data.LBXSGL > 500) {
    errors.push("La glucosa en ayunas debe estar entre 30 y 500 mg/dL.");
  }
  if (!data._raw.triglyceridesRaw || data._raw.triglyceridesRaw < 20 || data._raw.triglyceridesRaw > 1000) {
    errors.push("Los triglicéridos deben estar entre 20 y 1000 mg/dL.");
  }
  if (!data.LBDHDD || data.LBDHDD < 10 || data.LBDHDD > 150) {
    errors.push("El colesterol HDL debe estar entre 10 y 150 mg/dL.");
  }
  if (!data.BPXSY1 || data.BPXSY1 < 60 || data.BPXSY1 > 250) {
    errors.push("La presión sistólica debe estar entre 60 y 250 mmHg.");
  }
  if (!data.BPXDI1 || data.BPXDI1 < 30 || data.BPXDI1 > 150) {
    errors.push("La presión diastólica debe estar entre 30 y 150 mmHg.");
  }

  return { valid: errors.length === 0, errors };
}

// ─── API / ML Prediction Function ──────────────
/**
 * Sends user health data to the FastAPI ML prediction endpoint.
 *
 * @param {Object} userData - Object with NHANES variable names
 * @returns {Promise<Object>} Prediction result from backend
 */
async function fetchMLPrediction(userData) {
  // Build the API body (exclude _raw metadata)
  const { _raw, ...apiPayload } = userData;

  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify(apiPayload),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Error del servidor: ${response.status}`);
  }

  return await response.json();
}

// ─── UI State Management ───────────────────────
/**
 * Transitions the results panel to a specific state.
 * @param {"idle" | "loading" | "result"} state
 */
function setResultsState(state) {
  DOM.resultsIdle.style.display = state === "idle" ? "flex" : "none";
  DOM.resultsLoading.style.display = state === "loading" ? "flex" : "none";
  DOM.resultsLoading.classList.toggle("is-active", state === "loading");
  DOM.resultsCard.style.display = state === "result" ? "block" : "none";
  DOM.resultsCard.classList.toggle("is-active", state === "result");
}

/**
 * Maps the API risk_level to display info.
 */
function getRiskDisplay(riskLevel) {
  const map = {
    low: { label: "Normal", category: "low" },
    moderate: { label: "Prediabetes", category: "moderate" },
    high: { label: "Diabetes", category: "high" },
  };
  return map[riskLevel] || map.low;
}

/**
 * Renders prediction results from the API into the results card.
 * @param {Object} apiResult - Response from /predict endpoint
 * @param {Object} rawData - Raw form values for display
 */
function renderResults(apiResult, rawData) {
  const riskDisplay = getRiskDisplay(apiResult.risk_level);

  // Risk circle
  DOM.riskCircle.setAttribute("data-risk", riskDisplay.category);

  // Show probability as percentage
  const mainProba = apiResult.risk_level === "high"
    ? apiResult.probability_diabetes
    : (apiResult.probability_prediabetes ?? apiResult.probability_diabetes);
  DOM.riskValue.textContent = `${Math.round(mainProba * 100)}%`;

  // Text
  DOM.riskLevel.textContent = apiResult.prediction;
  DOM.riskDescription.textContent = apiResult.description;

  // Metrics (display raw user values)
  DOM.metricBmi.textContent = rawData.bmi.toFixed(1);
  DOM.metricGlucose.textContent = document.getElementById("fieldGlucose").value;
  DOM.metricTriglycerides.textContent = document.getElementById("fieldTriglycerides").value;
  DOM.metricHdl.textContent = document.getElementById("fieldHdl").value;
  DOM.metricSystolic.textContent = document.getElementById("fieldSystolic").value;
  DOM.metricDiastolic.textContent = document.getElementById("fieldDiastolic").value;

  // Timestamp
  DOM.resultTimestamp.textContent = new Date().toLocaleString("es-ES", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  setResultsState("result");
}

// ─── Event Handlers ────────────────────────────
/**
 * Handles form submission.
 * @param {Event} e
 */
async function handleSubmit(e) {
  e.preventDefault();

  const userData = collectFormData();
  const validation = validateFormData(userData);

  if (!validation.valid) {
    alert(validation.errors.join("\n"));
    return;
  }

  // Show loading state
  DOM.submitBtn.disabled = true;
  setResultsState("loading");

  // Smooth scroll to results
  document.getElementById("results").scrollIntoView({
    behavior: "smooth",
    block: "center",
  });

  try {
    const apiResult = await fetchMLPrediction(userData);
    renderResults(apiResult, userData._raw);
  } catch (error) {
    console.error("Prediction error:", error);
    setResultsState("idle");
    alert(`Error: ${error.message}\n\nAsegúrate de que el servidor API esté corriendo en ${API_URL}`);
  } finally {
    DOM.submitBtn.disabled = false;
  }
}

/**
 * Resets form and results to initial state.
 */
function handleReset() {
  DOM.form.reset();
  setResultsState("idle");
}

// ─── Scroll Reveal Animation ───────────────────
function initScrollReveal() {
  const revealElements = document.querySelectorAll(".reveal");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.15,
      rootMargin: "0px 0px -40px 0px",
    }
  );

  revealElements.forEach((el) => observer.observe(el));
}

// ─── Animated Stats Counter ────────────────────
function initStatsCounter() {
  const statNumbers = document.querySelectorAll(".stats__number");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const target = parseInt(el.getAttribute("data-target"), 10);
          const suffix = el.getAttribute("data-suffix") || "";
          animateCounter(el, 0, target, 1200, suffix);
          observer.unobserve(el);
        }
      });
    },
    { threshold: 0.5 }
  );

  statNumbers.forEach((el) => observer.observe(el));
}

/**
 * Animates a number from start to end with easing.
 */
function animateCounter(el, start, end, duration, suffix) {
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.floor(start + (end - start) * eased);
    el.textContent = current.toLocaleString("es") + suffix;

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

// ─── Initialize ────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Bind form events
  DOM.form.addEventListener("submit", handleSubmit);
  DOM.resetBtn.addEventListener("click", handleReset);

  // Init animations
  initScrollReveal();
  initStatsCounter();

  // Set initial state
  setResultsState("idle");
});
