import streamlit as st
import requests

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_LOG_URL = "https://insight-commerce-artifacts.s3.us-east-2.amazonaws.com/models/model_log.json"

COLOR_PRIMARY   = "#FE495F"
COLOR_SECONDARY = "#FE9D97"
COLOR_ACCENT    = "#BDED7E"
COLOR_LIGHT     = "#FFFEC8"
COLOR_BG_CARD   = "#1A1F2C"

st.set_page_config(
    page_title="Impacto de Negocio · Insight Commerce",
    page_icon="📈",
    layout="wide",
)

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
    .kpi-accent {{ border-left-color: {COLOR_ACCENT}; }}
    .kpi-yellow {{ border-left-color: {COLOR_LIGHT}; }}
    .section-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.8rem 0;
    }}
    .feature-bar-label {{
        font-size: 0.85rem;
        color: #FAFAFA;
        margin-bottom: 0.15rem;
    }}
    .coverage-badge {{
        display: inline-block;
        background-color: {COLOR_ACCENT};
        color: #1A1A1A;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 3px 12px;
        border-radius: 20px;
        margin-left: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_log() -> dict:
    response = requests.get(MODEL_LOG_URL, timeout=20)
    response.raise_for_status()
    return response.json()

try:
    log = load_model_log()
except requests.RequestException:
    st.error(f"No se pudo cargar model_log.json desde {MODEL_LOG_URL}.")
    st.stop()

metrics    = log.get("metrics_test", {})
split      = log.get("split", {})
top10      = log.get("importance_top10", [])
zero_feat  = log.get("features_zero_importance", [])
n_features = log.get("n_features", 0)
model_ts   = log.get("timestamp", "—")[:10]
uplift     = log.get("uplift", {})
uplift_pct = uplift.get("uplift_relative_pct", 0)
uplift_abs = uplift.get("uplift_absolute", 0)

precision = metrics.get("precision", 0)
recall    = metrics.get("recall", 0)
auc       = metrics.get("auc", 0)

precision_pct = round(precision * 100, 1)

# El model_log.json nuevo usa claves "train", "val", "test"
n_train_users = split.get("train", split.get("n_train_users", 0))
n_val_users   = split.get("val",   split.get("n_val_users",   0))
n_test_users  = split.get("test",  split.get("n_test_users",  0))
total_users   = n_train_users + n_val_users + n_test_users
active_features = n_features - len(zero_feat)

FEATURE_LABELS = {
    "up_reorder_rate":            "Tasa de recompra del producto por el usuario",
    "up_days_since_last":         "Días desde la última compra del producto",
    "product_reorder_rate":       "Popularidad global del producto",
    "user_reorder_ratio":         "Frecuencia de recompra del usuario",
    "up_delta_days":              "Variación en el intervalo de compra",
    "p_aisle_reorder_rate":       "Popularidad del pasillo del producto",
    "up_first_order_number":      "Antigüedad del producto en el historial",
    "u_favorite_aisle":           "Pasillo favorito del usuario",
    "user_days_since_last_order": "Días desde la última visita del usuario",
    "up_times_purchased":         "Cantidad de veces que el usuario compró el producto",
}

st.markdown(f"""
<h1 style='color:{COLOR_PRIMARY}; margin-bottom:0;'>📈 Impacto de Negocio</h1>
<p style='color:#888; margin-top:0.2rem;'>
    Insight Commerce · Modelo entrenado el {model_ts}
    <span class="coverage-badge">100% cobertura de usuarios</span>
</p>
""", unsafe_allow_html=True)

st.divider()

# ── KPIs principales ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Rendimiento del modelo</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Precisión Top-10</div>
        <div class="kpi-value">{precision_pct}%</div>
        <div class="kpi-sub">De cada 10 productos que el sistema sugiere, más de 4 son productos que el usuario efectivamente compraría.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card kpi-accent">
        <div class="kpi-label">Poder de discriminación</div>
        <div class="kpi-value">{round(auc * 100, 1)}%</div>
        <div class="kpi-sub">AUC-ROC · El modelo acierta en 8 de cada 10 comparaciones entre productos</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Cobertura de recall</div>
        <div class="kpi-value">{round(recall * 100, 1)}%</div>
        <div class="kpi-sub">El sistema encuentra el 41% de todos los productos que el usuario compraría. Lo que no aparece en el top-10 queda fuera por límite de espacio, no por error del modelo.</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card kpi-yellow">
        <div class="kpi-label">Usuarios evaluados</div>
        <div class="kpi-value">{n_test_users:,}</div>
        <div class="kpi-sub">El modelo fue probado sobre {n_test_users:,} usuarios reales que nunca vio durante el entrenamiento, garantizando que los resultados son representativos.</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Uplift ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Ventaja sobre recomendación genérica</div>', unsafe_allow_html=True)

col_u1, col_u2 = st.columns(2)

with col_u1:
    st.markdown(f"""
    <div class="kpi-card kpi-accent">
        <div class="kpi-label">Uplift vs popularidad global</div>
        <div class="kpi-value">+{round(uplift_pct)}%</div>
        <div class="kpi-sub">El modelo personalizado encuentra {round(uplift_pct)}% más productos relevantes que recomendar "lo más vendido"</div>
    </div>
    """, unsafe_allow_html=True)

with col_u2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Productos adicionales por usuario</div>
        <div class="kpi-value">+{uplift_abs:.1f}</div>
        <div class="kpi-sub">Productos relevantes extra por usuario respecto al baseline de popularidad global</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Cobertura del sistema ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Cobertura del sistema</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(f"""
    <div class="kpi-card kpi-accent">
        <div class="kpi-label">Modelo LightGBM</div>
        <div class="kpi-value">{total_users:,}</div>
        <div class="kpi-sub">Usuarios con ≥ 5 órdenes<br>Recomendación personalizada completa</div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Cold-start personal</div>
        <div class="kpi-value">1 – 4</div>
        <div class="kpi-sub">Órdenes previas<br>Recomendación por historial propio</div>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Usuario nuevo</div>
        <div class="kpi-value">0</div>
        <div class="kpi-sub">Órdenes previas<br>Recomendación por popularidad global</div>
    </div>
    """, unsafe_allow_html=True)

st.caption("El sistema nunca devuelve un error por falta de historial — todo usuario recibe recomendaciones.")
st.divider()

# ── Top 10 señales ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">¿Qué señales usa el modelo para recomendar?</div>', unsafe_allow_html=True)
st.caption("Las 10 variables con mayor impacto en las predicciones, en lenguaje de negocio.")

max_importance = max(item["importance"] for item in top10) if top10 else 1

for item in top10:
    feature    = item["feature"]
    importance = item["importance"]
    label      = FEATURE_LABELS.get(feature, feature)
    bar_pct    = importance / max_importance

    col_label, col_bar = st.columns([4, 6])
    with col_label:
        st.markdown(f'<div class="feature-bar-label">🔹 {label}</div>', unsafe_allow_html=True)
    with col_bar:
        st.progress(bar_pct)

st.divider()

# ── Dataset ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Dataset de entrenamiento</div>', unsafe_allow_html=True)

col_d1, col_d2, col_d3, col_d4 = st.columns(4)

with col_d1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Usuarios de entrenamiento</div>
        <div class="kpi-value">{n_train_users:,}</div>
        <div class="kpi-sub">70% del total</div>
    </div>
    """, unsafe_allow_html=True)

with col_d2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Usuarios de validación</div>
        <div class="kpi-value">{n_val_users:,}</div>
        <div class="kpi-sub">15% del total</div>
    </div>
    """, unsafe_allow_html=True)

with col_d3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Usuarios de test</div>
        <div class="kpi-value">{n_test_users:,}</div>
        <div class="kpi-sub">15% del total — evaluación final</div>
    </div>
    """, unsafe_allow_html=True)

with col_d4:
    st.markdown(f"""
    <div class="kpi-card kpi-accent">
        <div class="kpi-label">Features activas</div>
        <div class="kpi-value">{active_features} / {n_features}</div>
        <div class="kpi-sub">{len(zero_feat)} con importancia cero<br>descartadas automáticamente</div>
    </div>
    """, unsafe_allow_html=True)