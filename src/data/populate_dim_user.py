"""
populate_dim_user.py
Puebla las columnas user_name, user_address y user_birthdate de dim_user
en Neon con datos sintéticos generados con Faker.

Ejecutar una sola vez después del ETL:
    python src/data/populate_dim_user.py

Requisitos:
    pip install faker psycopg2-binary python-dotenv
"""

import psycopg2
import os
import logging
from dotenv import load_dotenv
from faker import Faker
from psycopg2.extras import execute_values

# ── Configuración de logs ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/logs/populate_dim_user.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

# ── Credenciales Neon ──────────────────────────────────────────────────────────
NEON_DB_CONFIG = {
    'host'    : os.getenv('NEON_HOST'),
    'database': os.getenv('NEON_DATABASE'),
    'user'    : os.getenv('NEON_USER'),
    'password': os.getenv('NEON_PASSWORD'),
    'port'    : os.getenv('NEON_PORT', '5432'),
    'sslmode' : os.getenv('NEON_SSLMODE', 'require'),
}

# ── Parámetros ─────────────────────────────────────────────────────────────────
RANDOM_SEED  = int(os.getenv('RANDOM_SEED', 42))
BATCH_SIZE   = 1_000
FAKER_LOCALE = 'en_US'


def populate_dim_user():
    fake = Faker(FAKER_LOCALE)
    Faker.seed(RANDOM_SEED)

    try:
        conn = psycopg2.connect(**NEON_DB_CONFIG)
        cur  = conn.cursor()
        logging.info("Conexión a Neon exitosa.")

        # ── Obtener user_keys que tienen NULL en user_name ─────────────────────
        cur.execute("""
            SELECT user_key
            FROM dim_user
            WHERE user_name IS NULL
            ORDER BY user_key
        """)
        user_keys = [row[0] for row in cur.fetchall()]
        logging.info(f"Usuarios a poblar: {len(user_keys):,}")

        if not user_keys:
            logging.info("Todos los usuarios ya tienen datos sintéticos. Nada que hacer.")
            conn.close()
            return

        # ── Generar datos sintéticos ───────────────────────────────────────────
        records = []
        for user_key in user_keys:
            records.append((
                fake.name(),
                fake.address().replace('\n', ', '),
                fake.date_of_birth(minimum_age=18, maximum_age=80),
                user_key
            ))

        # ── Actualizar en batches ──────────────────────────────────────────────
        total_updated = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            execute_values(
                cur,
                """
                UPDATE dim_user AS d SET
                    user_name      = v.user_name,
                    user_address   = v.user_address,
                    user_birthdate = v.user_birthdate
                FROM (VALUES %s) AS v(user_name, user_address, user_birthdate, user_key)
                WHERE d.user_key = v.user_key
                """,
                batch,
                template="(%s, %s, %s::date, %s)"
            )
            conn.commit()
            total_updated += len(batch)
            logging.info(f"Actualizados: {total_updated:,} / {len(records):,}")

        logging.info(f"Completado. {total_updated:,} usuarios poblados con datos sintéticos.")
        conn.close()

    except Exception as e:
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    populate_dim_user()
