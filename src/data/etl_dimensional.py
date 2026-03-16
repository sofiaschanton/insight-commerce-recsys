import psycopg2
import logging
import os
from faker import Faker
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from datetime import datetime

# ── Configuración de logs ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/logs/dimensional_etl.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

# ── Credenciales base de datos local ──────────────────────────────────────────
LOCAL_DB_CONFIG = {
    'host'    : os.getenv('LOCAL_HOST'),
    'database': os.getenv('LOCAL_DATABASE'),
    'user'    : os.getenv('LOCAL_USER'),
    'password': os.getenv('LOCAL_PASSWORD'),
    'port'    : os.getenv('LOCAL_PORT', '5432'),
}

# ── Credenciales AWS ──────────────────────────────────────────────────────────
# Credenciales en .env — nunca hardcodear ni subir al repositorio.
# Variables requeridas: AWS_HOST, AWS_DATABASE, AWS_USER, AWS_PASSWORD,
# AWS_PORT, AWS_SSLMODE
AWS_DB_CONFIG = {
    'host'             : os.getenv('AWS_HOST'),
    'database'         : os.getenv('AWS_DATABASE'),
    'user'             : os.getenv('AWS_USER'),
    'password'         : os.getenv('AWS_PASSWORD'),
    'port'             : os.getenv('AWS_PORT', '5432'),
    'sslmode'          : os.getenv('AWS_SSLMODE'),
}

# ── Parámetros del pipeline ────────────────────────────────────────────────────
# N_USERS_APTOS: cantidad de usuarios aptos a cargar en Amazon RDS.
# Un usuario apto cumple: >= 5 órdenes prior + exactamente 1 orden train.
# Valor acordado para AWS free tier (20 GB): 10.000 usuarios aptos.
N_USERS_APTOS    = 10_000
MIN_USER_ORDERS  = 5    # Feature Schema v6.0
MIN_PRODUCT_ORDERS = 50 # Feature Schema v6.0 — EDA Sección 3
RANDOM_SEED  = int(os.getenv('RANDOM_SEED', 42))
BATCH_SIZE   = 1_000
FAKER_LOCALE = 'en_US'

class DimensionalETL:
    def __init__(self, local_config: dict, aws_config: dict):
        """
        Inicializa el ETL con las configuraciones de conexión.
        Args:
            local_config (dict): Credenciales de la base de datos PostgreSQL local.
            aws_config  (dict): Credenciales de la base de datos AWS.
        """
        self.local_config      = local_config
        self.aws_config        = aws_config
        self.local_conn        = None
        self.aws_conn         = None
        self.batch_id          = int(datetime.now().timestamp())
        self.report_stats      = {}
        self.pipeline_start_time = None

    def connect(self) -> bool:
        """
        Establece conexiones a la base de datos local y Amazon RDS.

        Returns:
            bool: True si ambas conexiones son exitosas, False en caso de error.
        """
        try:
            self.local_conn = psycopg2.connect(**self.local_config)
            self.aws_conn  = psycopg2.connect(**self.aws_config)
            logging.info("Conexiones a Local y Amazon RDS exitosas.")
            return True
        except Exception as e:
            logging.error(f"Error al conectar a las bases de datos: {e}")
            return False

    def close(self):
        """Cierra de forma segura las conexiones activas."""
        if self.local_conn:
            self.local_conn.close()
        if self.aws_conn:
            self.aws_conn.close()
        logging.info("Conexiones cerradas.")

    def transfer_data(
        self,
        extract_query: str,
        target_table: str,
        insert_query: str,
        chunk_size: int = 10_000
    ):
        """
        Extrae datos de la base de datos local usando cursor del lado del servidor
        e inserta en Amazon RDS por lotes para optimizar uso de RAM y evitar timeouts.

        Args:
            extract_query (str): SQL SELECT para extraer datos del origen local.
            target_table  (str): Nombre de la tabla destino en AWS RDS.
            insert_query  (str): SQL INSERT parametrizado para el destino.
            chunk_size    (int): Registros por lote. Por defecto 10.000.
        """
        logging.info(f"Iniciando transferencia: {target_table}")
        start_time     = datetime.now()
        total_inserted = 0
        status         = "Completado"
        error_msg      = "Ninguno"

        try:
            local_cursor = self.local_conn.cursor(name=f'fetch_cursor_{target_table}')
            local_cursor.execute(extract_query)
            aws_cursor = self.aws_conn.cursor()

            while True:
                records = local_cursor.fetchmany(chunk_size)
                if not records:
                    break
                execute_values(aws_cursor, insert_query, records, page_size=chunk_size)
                self.aws_conn.commit()
                total_inserted += len(records)
                logging.info(f"{target_table}: {total_inserted:,} registros transferidos...")

            local_cursor.close()
            aws_cursor.close()
            logging.info(f"Transferencia de {target_table} completada. Total: {total_inserted:,}")

        except Exception as e:
            self.aws_conn.rollback()
            status    = "Fallido"
            error_msg = str(e)
            logging.error(f"Error transfiriendo {target_table}: {e}")

        finally:
            duration = datetime.now() - start_time
            self.report_stats[target_table] = {
                "estado"          : status,
                "filas_procesadas": total_inserted,
                "duracion"        : duration,
                "error"           : error_msg,
            }

    def populate_dim_user(self):
        """
        Puebla las columnas user_name, user_address y user_birthdate de dim_user
        en Amazon RDS con datos sintéticos generados con Faker.

        Ejecutar una sola vez después de la subida de las tablas

        Requisitos:
            pip install faker psycopg2-binary python-dotenv
        """
        fake = Faker(FAKER_LOCALE)
        Faker.seed(RANDOM_SEED)

        try:
            conn = psycopg2.connect(**AWS_DB_CONFIG)
            cur  = conn.cursor()
            logging.info("Conexión a Amazon RDS exitosa.")

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
                return

            records = []
            for user_key in user_keys:
                records.append((
                    fake.name(),
                    fake.address().replace('\n', ', '),
                    fake.date_of_birth(minimum_age=18, maximum_age=80),
                    user_key
                ))

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

        except Exception as e:
            logging.error(f"Error en populate_dim_user: {e}")
            raise

    def generate_report(self):
        """
        Genera un reporte estructurado en logs con métricas por tabla
        y totales del pipeline. 
        
        """
        if not self.pipeline_start_time:
            logging.warning("El pipeline no fue iniciado correctamente.")
            return

        total_duration = datetime.now() - self.pipeline_start_time
        total_rows     = sum(s['filas_procesadas'] for s in self.report_stats.values())

        report_lines = [
            "",
            "=" * 75,
            "  REPORTE DE EJECUCION - ETL Dimensional - Instacart Dataset",
            "=" * 75,
            f"Batch ID        : {self.batch_id}",
            f"Hora de Inicio  : {self.pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Tiempo Total    : {total_duration}",
            f"Total Filas     : {total_rows:,}",
            f"N_USERS_APTOS   : {N_USERS_APTOS:,}",
            f"MIN_USER_ORDERS : {MIN_USER_ORDERS}",
            f"MIN_PROD_ORDERS : {MIN_PRODUCT_ORDERS}",
            f"RANDOM_SEED     : {RANDOM_SEED}",
            "-" * 75,
            f"{'TABLA DESTINO':<25} | {'ESTADO':<12} | {'FILAS':<10} | {'DURACIÓN':<15}",
            "-" * 75,
        ]

        for table, stats in self.report_stats.items():
            dur_str = str(stats['duracion']).split('.')[0]
            report_lines.append(
                f"{table:<25} | {stats['estado']:<12} | {stats['filas_procesadas']:<10,} | {dur_str:<15}"
            )
            if stats['estado'] == 'Fallido':
                report_lines.append(f"  -> ERROR: {stats['error']}")

        report_lines.append("=" * 75)
        report_lines.append("")
        logging.info("\n".join(report_lines))

    def run_pipeline(self):
        """
        Orquesta la ejecución completa del pipeline ETL.
        Flujo:
          1. Registrar tiempo de inicio.
          2. Conectar a las bases de datos.
          3. Fijar semilla en PostgreSQL local para reproducibilidad.
          4. Transferir dim_user → dim_product → fact_order_products.
          5. Poblar datos sintéticos en dim_user.
          6. Cerrar conexiones.
          7. Generar reporte.
        Filtros aplicados (Feature Schema v6.0 + EDA Sección 6.4):
          - eval_set != 'test'
          - Usuarios con >= 5 órdenes en eval_set='prior'
          - Usuarios con >= 1 orden en eval_set='train'
          - Productos con >= 50 compras globales en prior
          - LIMIT N_USERS_APTOS = 10.000 usuarios aptos
          - RANDOM_SEED fijado en PostgreSQL local antes del SELECT para reproducibilidad
        """
        self.pipeline_start_time = datetime.now()

        if not self.connect():
            return

        # ── Fijar semilla en PostgreSQL local ──────────────────────────────────
        # setseed() acepta un valor entre -1 y 1.
        # RANDOM_SEED=42 → 42/100 = 0.42, dentro del rango válido.
        # Esto garantiza que ORDER BY RANDOM() sea reproducible entre corridas.
        seed_value = RANDOM_SEED / 100
        seed_cursor = self.local_conn.cursor()
        seed_cursor.execute(f"SELECT setseed({seed_value})")
        seed_cursor.close()
        self.local_conn.commit()
        logging.info(f"Semilla fijada en PostgreSQL local: setseed({seed_value})")

        # ── 1. DIM_USER ────────────────────────────────────────────────────────
        # Conserva user_id original de Instacart como user_key.
        # fact_order_products también usa user_id original → JOINs consistentes.
        # Filtro: >= 5 órdenes prior AND >= 1 orden train (usuarios aptos).
        # ORDER BY RANDOM() con semilla fija → selección reproducible.
        query_ext_user = f"""
            SELECT
                u.user_id AS user_key,
                u.user_name,
                u.user_address,
                DATE_TRUNC('year', CURRENT_DATE
                    - MAKE_INTERVAL(years := u.user_age))::DATE AS user_birthdate
            FROM users_Schema.users u
            WHERE u.user_id IN (
                SELECT user_id
                FROM orders_schema.orders
                GROUP BY user_id
                HAVING
                    COUNT(order_id) FILTER (WHERE eval_set = 'prior') >= {MIN_USER_ORDERS}
                    AND COUNT(order_id) FILTER (WHERE eval_set = 'train') >= 1
                ORDER BY RANDOM()
                LIMIT {N_USERS_APTOS}
            );
        """

        query_ins_user = """
            INSERT INTO dim_user (user_key, user_name, user_address, user_birthdate)
            VALUES %s
            ON CONFLICT (user_key) DO NOTHING;
        """

        self.transfer_data(query_ext_user, 'dim_user', query_ins_user)

        # ── 2. DIM_PRODUCT ─────────────────────────────────────────────────────
        # Desnormalización de productos, aisles y departamentos.
        # Filtro: solo productos con >= 50 compras globales en prior.
        query_ext_product = f"""
            SELECT
                p.product_id   AS product_key,
                p.product_name,
                a.aisle        AS aisle_name,
                d.department   AS department_name
            FROM Resources.products p
            JOIN Resources.aisles        a ON p.aisle_id      = a.aisle_id
            JOIN Departments.departments d ON p.department_id = d.department_id
            WHERE p.product_id IN (
                SELECT product_id
                FROM Orders_Schema.order_products_prior
                GROUP BY product_id
                HAVING COUNT(*) >= {MIN_PRODUCT_ORDERS}
            );
        """

        query_ins_product = """
            INSERT INTO dim_product (product_key, product_name, aisle_name, department_name)
            VALUES %s
            ON CONFLICT (product_key) DO NOTHING;
        """

        self.transfer_data(query_ext_product, 'dim_product', query_ins_product)

        # ── 3. FACT_ORDER_PRODUCTS ─────────────────────────────────────────────
        # Unión de prior + train con filtros de usuarios y productos aptos.
        # eval_set='test' excluido por el WHERE final.
        # Usuarios aptos: los cargados en dim_user (JOIN actúa como filtro).
        # Productos aptos: los cargados en dim_product (JOIN actúa como filtro).
        # Obtener los user_keys ya cargados en Amazon RDS 
        aws_cursor = self.aws_conn.cursor()
        aws_cursor.execute('SELECT user_key FROM dim_user')
        loaded_users = tuple(row[0] for row in aws_cursor.fetchall())
        aws_cursor.close()

        query_ext_fact = f"""
            WITH All_Orders AS (
                SELECT order_id, product_id, add_to_cart_order, reordered
                FROM orders_schema.order_products_prior
                UNION ALL
                SELECT order_id, product_id, add_to_cart_order, reordered
                FROM Orders_Schema.order_products_train
            ),
            Productos_Aptos AS (
                SELECT product_id
                FROM Orders_Schema.order_products_prior
                GROUP BY product_id
                HAVING COUNT(*) >= {MIN_PRODUCT_ORDERS}
            )
            SELECT
                o.order_id                         AS order_key,
                o.user_id                          AS user_key,
                op.product_id                      AS product_key,
                o.order_dow,
                CAST(o.order_hour_of_day AS SMALLINT),
                o.days_since_prior_order,
                op.add_to_cart_order,
                op.reordered,
                o.order_number,
                o.eval_set                         AS get_eval
            FROM Orders_Schema.orders o
            JOIN All_Orders      op ON o.order_id    = op.order_id
            JOIN Productos_Aptos pa ON op.product_id = pa.product_id
            WHERE o.eval_set IN ('prior', 'train')
            AND o.user_id IN {loaded_users};
        """

        query_ins_fact = """
            INSERT INTO fact_order_products (
                order_key, user_key, product_key, order_dow, order_hour_of_day,
                days_since_prior_order, add_to_cart_order, reordered,
                order_number, get_eval
            )
            VALUES %s
            ON CONFLICT (order_key, product_key) DO NOTHING;
        """

        self.transfer_data(query_ext_fact, 'fact_order_products', query_ins_fact, chunk_size=10_000)

        self.populate_dim_user()
        self.generate_report()
        self.close()


if __name__ == "__main__":
    etl = DimensionalETL(LOCAL_DB_CONFIG, AWS_DB_CONFIG)
    etl.run_pipeline()
