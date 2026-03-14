import os
import requests
import streamlit as st

# ── Configuración ────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Muestra de user_keys válidos para la demo (primeros 30 de la lista completa)
SAMPLE_USERS = [
    1, 13, 78, 95, 111, 117, 124, 197, 210, 294,
    329, 393, 403, 411, 426, 454, 469, 486, 495, 502,
    519, 520, 524, 525, 558, 567, 593, 608, 634, 746,
]

st.set_page_config(
    page_title="Insight Commerce",
    page_icon="🛒",
    layout="centered",
)

# ── Título ───────────────────────────────────────────────────────────────────
st.title("Insight Commerce")
st.caption("Sistema de recomendación Next Basket · Instacart dataset")

st.divider()

# ── Ejemplos de usuarios válidos ─────────────────────────────────────────────
with st.expander("Ver ejemplos de usuarios válidos"):
    st.markdown(
        "Los user_keys válidos corresponden a usuarios cargados en la base de datos. "
        "Podés usar cualquiera de estos para probar el sistema:"
    )
    cols = st.columns(6)
    for i, user_id in enumerate(SAMPLE_USERS):
        cols[i % 6].code(str(user_id))

st.divider()

# ── Input ────────────────────────────────────────────────────────────────────
user_id_input = st.text_input(
    label="User ID",
    placeholder="Ingresá un user_id (ej: 42)",
    help="Ingresá el ID numérico del usuario para obtener sus recomendaciones.",
)

recomendar = st.button("Recomendar", type="primary")

# ── Lógica principal ─────────────────────────────────────────────────────────
if recomendar:
    if not user_id_input.strip():
        st.warning("Por favor ingresá un user_id antes de continuar.")
    elif not user_id_input.strip().isdigit():
        st.error("El user_id debe ser un número entero.")
    else:
        user_id = int(user_id_input.strip())

        with st.spinner(f"Generando recomendaciones para el usuario {user_id}..."):
            try:
                response = requests.post(
                    f"{API_URL}/recommend/{user_id}",
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])

                    st.success(f"Top {len(recommendations)} recomendaciones para el usuario {user_id}")
                    st.divider()

                    for i, item in enumerate(recommendations, start=1):
                        product_name = item.get("product_name") or f"Producto {item['product_key']}"
                        probability = item.get("probability", 0.0)

                        with st.container():
                            col_rank, col_info = st.columns([1, 9])

                            with col_rank:
                                st.markdown(f"### {i}")

                            with col_info:
                                st.markdown(f"**{product_name}**")
                                st.progress(
                                    value=probability,
                                    text=f"Probabilidad de recompra: {probability:.1%}",
                                )

                        st.divider()

                elif response.status_code == 404:
                    st.warning(
                        f"El usuario {user_id} no tiene historial de compras suficiente "
                        "para generar recomendaciones personalizadas."
                    )

                elif response.status_code == 503:
                    st.error(
                        "No se pudo conectar a la base de datos. "
                        "Verificá que la API esté corriendo y que las credenciales sean correctas."
                    )

                else:
                    st.error(
                        f"Error inesperado (código {response.status_code}). "
                        "Intentá nuevamente o revisá los logs de la API."
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    f"No se pudo conectar a la API en {API_URL}. "
                    "Verificá que FastAPI esté corriendo con `uvicorn src.api.main:app --reload`."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "La API tardó demasiado en responder (timeout 30s). "
                    "Puede ser que la consulta a la base de datos esté lenta."
                )
