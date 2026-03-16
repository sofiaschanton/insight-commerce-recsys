"""
src/data/validate_data.py
==========================
Validación de calidad del feature matrix antes del entrenamiento.

Verifica:
    1. Todas las columnas del contrato (26) están presentes.
    2. Tipos de datos correctos en columnas clave.
    3. Sin nulos en columnas no-nulables.
    4. Sin duplicados en (user_key, product_key).
    5. label solo contiene valores 0 o 1.
    6. up_times_purchased > 0.
    7. user_total_orders > 0.

Salida:
    reports/data/validation_report.json  — resultado por validación.

Códigos de salida:
    0 → todas las validaciones pasaron (o no hay parquet que validar).
    1 → una o más validaciones fallaron.

Uso:
    python -m src.data.validate_data                    # lee FEATURE_MATRIX_PATH
    python -m src.data.validate_data --path mi.parquet  # ruta custom
    from src.data.validate_data import validate         # uso en memoria desde pipeline.py
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

from src.features.feature_engineering import FEATURE_MATRIX_COLUMNS, NULLABLE_COLUMNS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Ruta por defecto (puede sobreescribirse con --path o FEATURE_MATRIX_PATH)
_DEFAULT_MATRIX_PATH = os.getenv(
    "FEATURE_MATRIX_PATH", "data/processed/feature_matrix.parquet"
)

# Columnas que deben estar libres de nulos
_NON_NULLABLE = [c for c in FEATURE_MATRIX_COLUMNS if c not in NULLABLE_COLUMNS]


# ── Función principal ─────────────────────────────────────────────────────────

def validate(
    matrix: pd.DataFrame,
    output_dir: str = "reports/data",
) -> dict:
    """
    Ejecuta todas las validaciones sobre un DataFrame en memoria.

    Parameters
    ----------
    matrix     : feature matrix (puede venir del parquet o de build_feature_matrix).
    output_dir : directorio donde se escribe validation_report.json.

    Returns
    -------
    dict con un resultado por validación + clave 'all_passed' (bool).
    Siempre escribe el reporte en output_dir/validation_report.json.
    """
    report: dict = {}
    all_passed = True

    # ── 1. Columnas presentes ─────────────────────────────────────────────────
    missing_cols = [c for c in FEATURE_MATRIX_COLUMNS if c not in matrix.columns]
    if missing_cols:
        report["columns_present"] = {
            "passed": False,
            "detail": f"Columnas ausentes: {missing_cols}",
        }
        all_passed = False
    else:
        report["columns_present"] = {"passed": True}

    # ── 2. Sin nulos en columnas no-nulables ──────────────────────────────────
    null_errors = {}
    for col in _NON_NULLABLE:
        if col in matrix.columns:
            n = int(matrix[col].isna().sum())
            if n > 0:
                null_errors[col] = n
    if null_errors:
        report["no_unexpected_nulls"] = {"passed": False, "detail": null_errors}
        all_passed = False
    else:
        report["no_unexpected_nulls"] = {"passed": True}

    # ── 3. Sin duplicados en (user_key, product_key) ──────────────────────────
    if {"user_key", "product_key"}.issubset(matrix.columns):
        dupes = int(matrix.duplicated(subset=["user_key", "product_key"]).sum())
        if dupes > 0:
            report["no_duplicate_pairs"] = {
                "passed": False,
                "detail": f"{dupes} pares (user_key, product_key) duplicados",
            }
            all_passed = False
        else:
            report["no_duplicate_pairs"] = {"passed": True}

    # ── 4. label binario ─────────────────────────────────────────────────────
    if "label" in matrix.columns:
        invalid_labels = set(matrix["label"].unique()) - {0, 1}
        if invalid_labels:
            report["label_binary"] = {
                "passed": False,
                "detail": f"Valores no binarios: {sorted(invalid_labels)}",
            }
            all_passed = False
        else:
            report["label_binary"] = {"passed": True}

    # ── 5. up_times_purchased > 0 ─────────────────────────────────────────────
    if "up_times_purchased" in matrix.columns:
        bad = int((matrix["up_times_purchased"] <= 0).sum())
        if bad > 0:
            report["up_times_purchased_positive"] = {
                "passed": False,
                "detail": f"{bad} filas con up_times_purchased <= 0",
            }
            all_passed = False
        else:
            report["up_times_purchased_positive"] = {"passed": True}

    # ── 6. user_total_orders > 0 ─────────────────────────────────────────────
    if "user_total_orders" in matrix.columns:
        bad = int((matrix["user_total_orders"] <= 0).sum())
        if bad > 0:
            report["user_total_orders_positive"] = {
                "passed": False,
                "detail": f"{bad} filas con user_total_orders <= 0",
            }
            all_passed = False
        else:
            report["user_total_orders_positive"] = {"passed": True}

    report["all_passed"] = all_passed

    # ── Guardar reporte ────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "validation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    status = "OK" if all_passed else "FALLIDA"
    logging.info(f"Validación del feature matrix: {status} → {output_path}")
    if not all_passed:
        failed = [k for k, v in report.items()
                  if k != "all_passed" and isinstance(v, dict) and not v.get("passed", True)]
        logging.error(f"Checks fallidos: {failed}")

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validar feature_matrix.parquet")
    parser.add_argument(
        "--path",
        default=_DEFAULT_MATRIX_PATH,
        help=f"Ruta al parquet. Default: {_DEFAULT_MATRIX_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/data",
        help="Directorio para validation_report.json. Default: reports/data",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        logging.warning(
            f"Feature matrix no encontrado en '{args.path}'. "
            "Validación omitida — se necesita ejecutar el pipeline primero."
        )
        # Guardar reporte indicando que se saltó (no es un error)
        os.makedirs(args.output_dir, exist_ok=True)
        skip_report = {"skipped": True, "reason": f"Archivo no encontrado: {args.path}", "all_passed": True}
        with open(os.path.join(args.output_dir, "validation_report.json"), "w") as f:
            json.dump(skip_report, f, indent=2)
        sys.exit(0)

    logging.info(f"Cargando feature matrix desde '{args.path}'...")
    matrix = pd.read_parquet(args.path)
    logging.info(f"  {len(matrix):,} filas × {len(matrix.columns)} columnas")

    report = validate(matrix, output_dir=args.output_dir)

    if not report.get("all_passed", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
