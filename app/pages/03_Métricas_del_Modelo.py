import streamlit as st
import requests

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_LOG_URL = "https://insight-commerce-artifacts.s3.us-east-2.amazonaws.com/models/model_log.json"

# Paleta Insight Commerce
COLOR_PRIMARY   = "#FE495F"
COLOR_SECONDARY = "#FE9D97"
COLOR_ACCENT    = "#BDED7E"
COLOR_LIGHT     = "#FFFEC8"
COLOR_BG_CARD   = "#1A1F2C"

st.set_page_config(
    page_title="Métricas del Modelo · Insight Commerce",
    page_icon="🔬",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .kpi-card {{
        background-color: {COLOR_BG_CARD};
        border-left: 4px solid {COLOR_PRIMARY};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
    }}
    .kpi-label {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {COLOR_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: #FAFAFA;
        line-height: 1.1;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        color: #888;
        margin-top: 0.2rem;
    }}
    .kpi-accent {{
        border-left-color: {COLOR_ACCENT};
    }}
    .kpi-yellow {{
        border-left-color: {COLOR_LIGHT};
    }}
    .section-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.8rem 0;
    }}
    .param-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #2A2F3C;
        font-size: 0.88rem;
    }}
    .param-key {{
        color: {COLOR_SECONDARY};
        font-weight: 600;
    }}
    .param-val {{
        color: #000000;
        font-family: monospace;
    }}
    .feature-chip {{
        display: inline-block;
        background-color: {COLOR_BG_CARD};
        border: 1px solid {COLOR_ACCENT};
        color: {COLOR_ACCENT};
        font-size: 0.78rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin: 3px;
    }}
    .feature-chip-zero {{
        border-color: #444;
        color: #666;
    }}
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def load_model_log() -> dict:
    response = requests.get(MODEL_LOG_URL, timeout=20)
    response.raise_for_status()
    return response.json()

try:
    log = load_model_log()
except requests.RequestException:
    st.error(
        f"No se pudo cargar model_log.json desde {MODEL_LOG_URL}. "
        "Verificá la conectividad y la disponibilidad del artefacto remoto."
    )
    st.stop()

metrics    = log.get("metrics_test", {})
split      = log.get("split", {})
top10      = log.get("importance_top10", [])
zero_feat  = log.get("features_zero_importance", [])
params     = log.get("best_params", {})
n_features = log.get("n_features", 0)
model_name = log.get("model_name", "LightGBM")
model_ts   = log.get("timestamp", "—")[:10]
all_features = log.get("feature_cols", [])
active_features = n_features - len(zero_feat)
active = [f for f in all_features if f not in zero_feat]

# ── Título ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='color:{COLOR_PRIMARY}; margin-bottom:0;'>🔬 Métricas del Modelo</h1>
<p style='color:#888; margin-top:0.2rem;'>
    {model_name} · Entrenado el {model_ts} · Split 70/15/15 por usuarios
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Métricas de evaluación ────────────────────────────────────────────────────
st.markdown('<div class="section-title">Evaluación sobre set de test</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Precision@10</div>
        <div class="kpi-value">{metrics.get('precision', 0):.4f}</div>
        <div class="kpi-sub">Proporción de productos relevantes<br>en el top-10 recomendado</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Recall@10</div>
        <div class="kpi-value">{metrics.get('recall', 0):.4f}</div>
        <div class="kpi-sub">Proporción de productos relevantes<br>efectivamente recuperados</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card kpi-accent">
        <div class="kpi-label">F1 Score</div>
        <div class="kpi-value">{metrics.get('f1', 0):.4f}</div>
        <div class="kpi-sub">Media armónica de Precision<br>y Recall</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card kpi-yellow">
        <div class="kpi-label">AUC-ROC</div>
        <div class="kpi-value">{metrics.get('auc', 0):.4f}</div>
        <div class="kpi-sub">Capacidad de discriminación<br>entre recompra y no recompra</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card kpi-accent">
        <div class="kpi-label">NDCG@10</div>
        <div class="kpi-value">{metrics.get('ndcg_at_10', 0):.4f}</div>
        <div class="kpi-sub">Calidad del ranking — considera<br>relevancia y posición de cada producto</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Split del dataset ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Split del dataset</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

splits = [
    ("Entrenamiento", "train", "70%", ""),
    ("Validación",    "val",   "15%", ""),
    ("Test",          "test",  "15%", "kpi-accent"),
]
for col, (label, u_key, pct, css) in zip([col_a, col_b, col_c], splits):
    with col:
        st.markdown(f"""
        <div class="kpi-card {css}">
            <div class="kpi-label">{label} · {pct}</div>
            <div class="kpi-value">{split.get(u_key, 0):,}</div>
            <div class="kpi-sub">usuarios</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Features ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Feature engineering</div>', unsafe_allow_html=True)

col_f1, col_f2 = st.columns([6, 4])

with col_f1:
    st.markdown(f"**Features activas ({active_features})**")
    active = [item["feature"] for item in top10]
    chips_active = " ".join(
        f'<span class="feature-chip">{f}</span>'
        for f in active
    )
    st.markdown(chips_active, unsafe_allow_html=True)

    st.markdown(f"<br>**Features con importancia cero ({len(zero_feat)}) — candidatas a eliminar**", unsafe_allow_html=True)
    chips_zero = " ".join(
        f'<span class="feature-chip feature-chip-zero">{f}</span>'
        for f in zero_feat
    )
    st.markdown(chips_zero, unsafe_allow_html=True)

with col_f2:
    st.markdown("**Top 10 por importancia**")
    max_imp = max(item["importance"] for item in top10) if top10 else 1
    for item in top10:
        pct = item["importance"] / max_imp
        st.markdown(
            f'<div style="font-size:0.8rem; color:#FAFAFA; margin-bottom:2px;">'
            f'{item["feature"]} <span style="color:{COLOR_SECONDARY};">({item["importance"]})</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(pct)

st.divider()

# ── Parámetros del modelo ─────────────────────────────────────────────────────
with st.expander("Parámetros óptimos — Optuna 50 trials"):
    PARAM_LABELS = {
        "n_estimators":       "Número de árboles",
        "learning_rate":      "Learning rate",
        "num_leaves":         "Hojas por árbol",
        "max_depth":          "Profundidad máxima",
        "min_child_samples":  "Muestras mínimas por hoja",
        "subsample":          "Subsample de filas",
        "colsample_bytree":   "Subsample de columnas",
        "reg_alpha":          "Regularización L1",
        "reg_lambda":         "Regularización L2",
        "scale_pos_weight":   "Peso clase positiva",
    }
    for key, val in params.items():
        if key in ("random_state", "n_jobs", "verbose"):
            continue
        label = PARAM_LABELS.get(key, key)
        if isinstance(val, float):
            if val < 0.0001:
                val_str = f"{val:.2e}"
            elif val < 0.001:
                val_str = f"{val:.6f}"
            else:
                val_str = f"{val:.4f}"
        else:
            val_str = str(val)
      
        st.markdown(
            f'<div class="param-row">'
            f'<span class="param-key">{label}</span>'
            f'<span class="param-val">{val_str}</span>'
            f'</div>',
            unsafe_allow_html=True
        )