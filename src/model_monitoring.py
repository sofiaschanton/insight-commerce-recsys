"""
src/model_monitoring.py
========================
Monitoreo de data drift usando PSI (Population Stability Index) y KS test.

Carga feature_matrix.parquet como distribución de referencia y la compara
contra un dataset actual para detectar drift en las features clave.

Si no existe un archivo de referencia separado, divide el parquet en dos mitades
(50% más antiguo = referencia, 50% más reciente = actual) para simular
el escenario de producción.

Umbrales de alerta:
    PSI < 0.10  → distribución estable
    PSI < 0.25  → cambio moderado (monitorear)
    PSI >= 0.25 → cambio significativo → evaluar reentrenamiento

    KS >= 0.30  → diferencia estadística significativa entre distribuciones
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Rutas configurables por variable de entorno
FEATURE_MATRIX_PATH = os.getenv(
    "FEATURE_MATRIX_PATH", "data/processed/feature_matrix.parquet"
)
REFERENCE_PATH = os.getenv(
    "REFERENCE_PATH", "data/processed/feature_matrix_reference.parquet"
)

# Features numéricas a monitorear — se excluyen NaN intencionales y columnas categóricas
MONITORED_FEATURES = [
    "user_total_orders",
    "user_avg_basket_size",
    "user_reorder_ratio",
    "product_total_purchases",
    "product_reorder_rate",
    "up_times_purchased",
    "up_reorder_rate",
    "up_days_since_last",
]


def _compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Computa el Population Stability Index entre dos distribuciones.

    PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

    Usa percentiles de 'expected' como breakpoints para que los bins sean
    equipopulados en la distribución de referencia.
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    eps = 1e-6
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = (expected_counts + eps) / len(expected)
    actual_pct = (actual_counts + eps) / len(actual)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def _compute_ks(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Computa el estadístico KS (Kolmogorov-Smirnov) entre dos distribuciones.

    KS = max|F_ref(x) - F_cur(x)|   donde F es la CDF empírica.
    No requiere scipy — usa numpy para construir las CDFs.
    """
    combined = np.concatenate([reference, current])
    combined_sorted = np.sort(np.unique(combined))

    cdf_ref = np.searchsorted(np.sort(reference), combined_sorted, side="right") / len(reference)
    cdf_cur = np.searchsorted(np.sort(current), combined_sorted, side="right") / len(current)

    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def _load_data(
    current_path: str,
    reference_path: str,
) -> tuple:
    """
    Carga los DataFrames de referencia y actual.

    Si no existe reference_path, loguea un warning y devuelve (None, None)
    para que compute_drift_metrics salga sin error.

    Returns
    -------
    tuple (df_ref, df_curr) o (None, None) si no hay parquet de referencia.
    """
    if not os.path.exists(current_path):
        logging.error(f"No se encontró feature_matrix en '{current_path}'.")
        sys.exit(1)

    if not os.path.exists(reference_path):
        logging.warning(
            f"No se encontró parquet de referencia en '{reference_path}'. "
            "Saltando monitoreo de drift — se necesitan al menos dos corridas del pipeline "
            "para comparar distribuciones. Ejecutar de nuevo tras el primer reentrenamiento."
        )
        return None, None

    df_ref = pd.read_parquet(reference_path)
    df_curr = pd.read_parquet(current_path)
    logging.info(
        f"Referencia cargada desde '{reference_path}' "
        f"({len(df_ref):,} filas). Actual: {len(df_curr):,} filas."
    )
    return df_ref, df_curr


def compute_drift_metrics(
    current_path: str = FEATURE_MATRIX_PATH,
    reference_path: str = REFERENCE_PATH,
    output_dir: str = ".",
) -> dict:
    """
    Calcula PSI y KS para cada feature monitoreada y genera drift_report.json.

    Parameters
    ----------
    current_path   : ruta al parquet con la distribución actual.
    reference_path : ruta al parquet de referencia (baseline de entrenamiento).
    output_dir     : directorio donde se escribe drift_report.json.

    Returns
    -------
    dict con psi, ks, drift_detected y breakdowns por feature.
    """
    logging.info("Iniciando escaneo de Data Drift (Métricas PSI y KS)...")

    df_ref, df_curr = _load_data(current_path, reference_path)

    if df_ref is None:
        logging.warning("Monitoreo omitido: no hay parquet de referencia disponible.")
        return {}

    available = [
        f for f in MONITORED_FEATURES
        if f in df_ref.columns and f in df_curr.columns
    ]
    if not available:
        logging.error(
            "Ninguna de las features monitoreadas existe en los datos. Abortando."
        )
        sys.exit(1)

    psi_by_feature = {}
    ks_by_feature = {}

    for feat in available:
        ref_vals = df_ref[feat].dropna().values.astype(float)
        cur_vals = df_curr[feat].dropna().values.astype(float)

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            logging.warning(f"  '{feat}': sin valores válidos, se omite.")
            continue

        psi_by_feature[feat] = round(_compute_psi(ref_vals, cur_vals), 4)
        ks_by_feature[feat] = round(_compute_ks(ref_vals, cur_vals), 4)

        logging.info(
            f"  {feat}: PSI={psi_by_feature[feat]:.4f}, KS={ks_by_feature[feat]:.4f}"
        )

    if not psi_by_feature:
        logging.error("No se pudieron calcular métricas para ninguna feature.")
        sys.exit(1)

    psi = round(float(np.mean(list(psi_by_feature.values()))), 3)
    ks = round(float(np.mean(list(ks_by_feature.values()))), 3)
    drift_detected = bool(psi > 0.25 or ks > 0.3)

    result = {
        "psi": psi,
        "ks": ks,
        "drift_detected": drift_detected,
        "psi_by_feature": psi_by_feature,
        "ks_by_feature": ks_by_feature,
    }

    if drift_detected:
        logging.warning(
            f"ALERTA DE DRIFT: métricas superan el umbral. PSI={psi}, KS={ks}"
        )
        logging.warning("Acción recomendada: evaluar reentrenamiento del modelo.")
    else:
        logging.info(
            f"Métricas estables. PSI={psi}, KS={ks}. No se requiere reentrenar."
        )

    output_path = os.path.join(output_dir, "drift_report.json")
    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"Reporte de monitoreo guardado en: {output_path}")
    except Exception as e:
        logging.error(f"Fallo crítico al intentar escribir el reporte: {e}")
        sys.exit(1)

    return result


if __name__ == "__main__":
    compute_drift_metrics()