import os
import time
import logging
import traceback
import pandas as pd
from typing import Optional, List, Tuple, Dict
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ─── Logging ──────────────────────────────────────────────────────────────────
# Usamos un logger nombrado (no el raíz) para no interferir con otros módulos.
# - Archivo (DEBUG): guarda todo, incluyendo los SQLs ejecutados → útil para debuggear
# - Consola (INFO): solo lo importante → filas cargadas, errores, resumen
logger = logging.getLogger('data_loader')
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    # Creamos el directorio de logs si no existe
    os.makedirs('./reports/logs', exist_ok=True)
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Handler de archivo: captura TODO (DEBUG+)
    fh = logging.FileHandler('./reports/logs/data_loader.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Handler de consola: captura solo INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ─── Env ──────────────────────────────────────────────────────────────────────
# Cargamos las variables de entorno desde .env
# Las credenciales de AWS RDS deben estar definidas como:
# AWS_USER, AWS_PASSWORD, AWS_HOST, AWS_PORT, AWS_DATABASE
load_dotenv()

def _get_engine():
    """
    Crea el engine de SQLAlchemy configurando correctamente SSL 
    para la conexión con AWS RDS.
    """
    # 1. Construir la URL base
    db_url = URL.create(
        drivername = 'postgresql+psycopg2',
        username   = os.getenv('AWS_USER'),
        password   = os.getenv('AWS_PASSWORD'),
        host       = os.getenv('AWS_HOST'),
        port       = int(os.getenv('AWS_PORT', '5432')),
        database   = os.getenv('AWS_DATABASE'),
    )

    # 2. Configurar argumentos de conexión (SSL)
    # Leemos el modo desde el .env (por defecto verify-full si queremos usar el certificado)
    ssl_mode = os.getenv("AWS_SSLMODE", "verify-full")
    
    connect_args = {}
    
    if ssl_mode in ["verify-full", "verify-ca"]:
        connect_args["sslmode"] = ssl_mode
        # Importante: El archivo debe estar en la raíz del proyecto
        connect_args["sslrootcert"] = "./global-bundle.pem"
        logger.info(f"Configurando conexión segura SSL (mode: {ssl_mode})")
    else:
        connect_args["sslmode"] = ssl_mode # Ej: 'require'

    # 3. Crear el engine con los argumentos de SSL
    return create_engine(db_url, connect_args=connect_args)

# ─── Configuración de tablas ──────────────────────────────────────────────────
# Tablas disponibles en el schema dimensional de AWS RDS
AVAILABLE_TABLES = ['fact_order_products', 'dim_user', 'dim_product']

# _TABLE_CONFIG define por cada tabla:
#   sql      → SELECT base sin filtros
#   user_col → columna para filtrar por usuario (None = tabla sin usuarios, se carga completa)
#   dtypes   → casteos de memoria para reducir RAM (ej: int64 → int32)
_TABLE_CONFIG: Dict[str, dict] = {
    'fact_order_products': {
        'sql': """
            SELECT order_key, user_key, product_key, order_number,
                   order_dow, order_hour_of_day, days_since_prior_order,
                   add_to_cart_order, reordered, get_eval
            FROM fact_order_products
        """,
        'user_col': 'user_key',
        'dtypes': {
            'order_key': 'int32', 'user_key': 'int32', 'product_key': 'int32',
            'order_number': 'int16', 'add_to_cart_order': 'int16', 'reordered': 'int8',
        },
    },
    'dim_user': {
        'sql': "SELECT user_key, user_name, user_address, user_birthdate FROM dim_user",
        'user_col': 'user_key',
        'dtypes': {'user_key': 'int32'},
    },
    'dim_product': {
        'sql': "SELECT product_key, product_name, aisle_name, department_name FROM dim_product",
        'user_col': None,  # sin user_key — siempre se carga completa
        'dtypes': {'product_key': 'int32'},
    },
}


def load_data_from_aws(
    tables: List[str] = AVAILABLE_TABLES,
    esquemas: Optional[List[str]] = None,
    n_users: Optional[int] = None,
    date_range: Optional[Tuple[str, str]] = None,
    connection_config: Optional[dict] = None,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Extrae datos desde AWS RDS (schema dimensional) en una sola carga.

    Parameters
    ----------
    tables : list[str]
        Tablas a extraer. Default: todas.
        Valores válidos: 'fact_order_products', 'dim_user', 'dim_product'.
    esquemas : list[str] | None
        Esquemas por tabla (mismo orden que tables). Ej: ['public', 'public'].
        None = usa el schema dimensional por defecto.
    n_users : int | None
        Cantidad de usuarios a samplear. None = todos.
        ⚠ Límite de usuarios únicos, no de filas.
    date_range : tuple[str, str] | None
        Rango ('YYYY-MM-DD', 'YYYY-MM-DD') para extracción incremental.
        Filtra por order_number como proxy temporal y propaga el filtro a dim_user.
    connection_config : dict | None
        Credenciales externas. Claves: 'AWS_HOST', 'AWS_DATABASE', 'AWS_USER',
        'AWS_PASSWORD', 'AWS_PORT'. None = usa .env.
    random_state : int
        Semilla para el sampleo. Default: 42.

    Returns
    -------
    dict[str, pd.DataFrame]

    Examples
    --------
    # Carga completa
    >>> data = load_data_from_aws()

    # Prueba rápida con 500 usuarios
    >>> data = load_data_from_aws(
    ...     tables=['fact_order_products', 'dim_user'],
    ...     n_users=500,
    ... )

    # Con filtro de fechas
    >>> data = load_data_from_aws(
    ...     tables=['fact_order_products', 'dim_user'],
    ...     n_users=1000,
    ...     date_range=('2017-01-01', '2017-06-30'),
    ... )

    # Esquemas custom
    >>> data = load_data_from_aws(
    ...     tables=['fact_order_products', 'dim_user'],
    ...     esquemas=['public', 'public'],
    ... )

    # Credenciales externas (sin .env)
    >>> data = load_data_from_aws(
    ...     tables=['fact_order_products'],
    ...     connection_config={
    ...         'AWS_HOST': 'aws_host',
    ...         'AWS_USER': 'postgres',
    ...         'AWS_PASSWORD': 'secret',
    ...         'AWS_DATABASE': 'aws_database',
    ...         'AWS_PORT': '5432',
    ...     }
    ... )
    """

    # ── Validación ─────────────────────────────────────────────────────────
    # Si no se usan esquemas custom, validamos que las tablas pedidas existan en _TABLE_CONFIG
    if not esquemas:
        invalid = [t for t in tables if t not in _TABLE_CONFIG]
        if invalid:
            raise ValueError(f"Tablas no reconocidas: {invalid}. Válidas: {AVAILABLE_TABLES}")

    # Si se usan esquemas custom, deben tener el mismo largo que tables
    if esquemas and len(esquemas) != len(tables):
        raise ValueError(
            f"'esquemas' debe tener el mismo largo que 'tables' ({len(esquemas)} != {len(tables)})."
        )
    

    # ── Conexión ───────────────────────────────────────────────────────────
    # Si se pasan credenciales externas se construye la URL manualmente,
    # sino se usa _get_engine() que lee del .env
    t0 = time.perf_counter()
    if connection_config:
        db_url = (
            f"postgresql+psycopg2://{connection_config.get('AWS_USER')}"
            f":{connection_config.get('AWS_PASSWORD')}"
            f"@{connection_config.get('AWS_HOST')}"
            f":{connection_config.get('AWS_PORT', '5432')}"
            f"/{connection_config.get('AWS_DATABASE', 'postgres')}"
        )
        logger.info("Usando connection_config externo")
        engine = create_engine(db_url)
    else:
        engine = _get_engine()
    load_log = []  # registro interno para el resumen final

    def _query(sql: str, label: str) -> pd.DataFrame:
        """
        Ejecuta un SQL y devuelve un DataFrame.
        Registra filas, MB y tiempo en load_log para el resumen final.
        El SQL se guarda en el archivo de log (nivel DEBUG) para facilitar el debugging.
        """
        t = time.perf_counter()
        status, n_rows, mem_mb = 'OK', 0, 0.0
        try:
            with engine.connect() as conn:  # la conexión se cierra automáticamente al salir del with
                df = pd.read_sql(text(sql), conn)
            n_rows = len(df)
            mem_mb = df.memory_usage(deep=True).sum() / 1e6
            logger.info(f"[{label}] {n_rows:>10,} filas | {mem_mb:.1f} MB | {time.perf_counter() - t:.2f}s")
            logger.debug(f"[{label}] SQL:\n{sql.strip()}")  # solo en archivo, no en consola
            return df
        except Exception as e:
            status = f"ERROR: {e}"
            logger.error(f"[{label}] Falló — {e}")
            logger.debug(traceback.format_exc())
            raise
        finally:
            # finally siempre se ejecuta, con éxito o con error
            load_log.append({
                'tabla': label, 'filas': n_rows,
                'mem_mb': round(mem_mb, 2),
                'tiempo_s': round(time.perf_counter() - t, 3),
                'estado': status,
            })

    logger.info(
        f"load_data_from_aws — tables={tables} | "
        f"n_users={n_users} | date_range={date_range}"
    )

    # ── Sampleo de usuarios ────────────────────────────────────────────────
    # Si n_users está definido, sampleamos N usuarios aleatorios de la fact table
    # y construimos un string SQL para usarlo como filtro IN en las queries siguientes
    user_ids_sql = ""
    if n_users is not None:
        # Intentamos fijar la semilla en Postgres para reproducibilidad del RANDOM()
        try:
            with engine.connect() as conn:
                pd.read_sql(text(f"SELECT setseed({random_state / (2**31 - 1)})"), conn)
        except Exception:
            pass  # si falla (ej: permisos) continuamos sin semilla fija
        df_users = _query(
            f"SELECT user_key FROM (SELECT DISTINCT user_key FROM fact_order_products) u ORDER BY RANDOM() LIMIT {n_users}",
            label='_sample_users'
        )
        user_ids_sql = ', '.join(str(u) for u in df_users['user_key'].tolist())
        logger.info(f"Filtrando por {len(df_users):,} usuarios")

    # ── Carga de tablas ────────────────────────────────────────────────────
    result = {}
    for i, tabla in enumerate(tables):
        if esquemas:
            # Modo esquemas custom: SELECT * con filtro de usuario si aplica
            where = f"WHERE user_key IN ({user_ids_sql})" if user_ids_sql else ""
            result[tabla] = _query(f"SELECT * FROM {esquemas[i]}.{tabla} {where}", label=tabla)
        else:
            # Modo normal: usamos el SQL y casteos definidos en _TABLE_CONFIG
            cfg  = _TABLE_CONFIG[tabla]
            ucol = cfg['user_col']
            # Solo filtramos por usuario si la tabla tiene user_col (dim_product no tiene)
            where = f"WHERE {ucol} IN ({user_ids_sql})" if user_ids_sql and ucol else ""
            df = _query(f"{cfg['sql'].strip()} {where}", label=tabla)
            # Casteamos columnas para reducir uso de RAM
            for col, dtype in cfg['dtypes'].items():
                df[col] = df[col].astype(dtype)
            result[tabla] = df

    # ── Filtro date_range ──────────────────────────────────────────────────
    # Como no hay columna de fecha real, usamos order_number como proxy temporal
    # mapeando el rango de fechas al rango de order_numbers proporcionalmente
    if date_range and 'fact_order_products' in result:
        from datetime import datetime
        start, end   = date_range
        df_fact      = result['fact_order_products']
        total_orders = df_fact['order_number'].max()
        # Rango de fechas asumido del dataset Instacart (2017)
        d_min, d_max = datetime(2017, 1, 1), datetime(2017, 12, 31)
        delta        = (d_max - d_min).days
        # Convertimos fechas a order_numbers equivalentes
        n_s = int(max(0, (datetime.strptime(start, '%Y-%m-%d') - d_min).days / delta) * total_orders)
        n_e = int(min(1, (datetime.strptime(end,   '%Y-%m-%d') - d_min).days / delta) * total_orders)

        before = len(df_fact)
        result['fact_order_products'] = df_fact[df_fact['order_number'].between(n_s, n_e)].copy()
        logger.info(
            f"date_range ({start} → {end}): {before:,} → "
            f"{len(result['fact_order_products']):,} filas [order_number {n_s}–{n_e}]"
        )

        # Propagamos el filtro a dim_user: solo mantenemos usuarios con órdenes en el rango
        if 'dim_user' in result:
            active   = result['fact_order_products']['user_key'].unique()
            before_u = len(result['dim_user'])
            result['dim_user'] = result['dim_user'][
                result['dim_user']['user_key'].isin(active)
            ].copy()
            logger.info(f"date_range → dim_user: {before_u:,} → {len(result['dim_user']):,} usuarios")

    # ── Resumen ────────────────────────────────────────────────────────────
    # Liberamos el pool de conexiones antes de devolver los datos
    engine.dispose()
    total_rows = sum(len(v) for v in result.values())
    elapsed    = time.perf_counter() - t0
    errors     = [r for r in load_log if r['estado'] != 'OK']

    logger.info("─" * 62)
    logger.info("RESUMEN DE SESIÓN")
    logger.info(f"  {'Tabla':<20} {'Filas':>10} {'MB':>7} {'Tiempo(s)':>10}  Estado")
    logger.info("─" * 62)
    for r in load_log:
        logger.info(
            f"  {r['tabla']:<20} {r['filas']:>10,} {r['mem_mb']:>7.1f} "
            f"{r['tiempo_s']:>10.3f}  {r['estado']}"
        )
    logger.info("─" * 62)
    logger.info(f"  {'TOTAL':<20} {total_rows:>10,} {'':>7} {elapsed:>10.3f}")
    if errors:
        logger.warning(f"  ⚠  {len(errors)} tabla(s) con error: {[e['tabla'] for e in errors]}")
    else:
        logger.info("  ✅ Todas las cargas exitosas")
    logger.info("─" * 62)

    return result

# ─── Guardado a disco ─────────────────────────────────────────────────────────
def save_data(
    data: Dict[str, pd.DataFrame],
    output_dir: str = 'data/raw',
) -> None:
    """
    Guarda cada DataFrame del dict como parquet en output_dir.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Resultado de load_data_from_aws().
    output_dir : str
        Directorio de salida. Se crea si no existe.

    Output
    ------
    data/raw/fact_order_products.parquet
    data/raw/dim_user.parquet
    data/raw/dim_product.parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, df in data.items():
        path = os.path.join(output_dir, f'{name}.parquet')
        df.to_parquet(path, index=False)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(f"  Guardado: {path} ({size_mb:.1f} MB)")


# ─── Entry point ──────────────────────────────────────────────────────────────
# python -m src.data.data_loader
# python -m src.data.data_loader --n_users 500
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carga de datos desde AWS RDS')
    parser.add_argument('--tables',  nargs='+', default=AVAILABLE_TABLES)
    parser.add_argument('--n_users', type=int,  default=None)
    parser.add_argument('--output',  type=str,  default='data/raw',
                        help='Directorio de salida para los parquet (default: data/raw)')
    args = parser.parse_args()

    data = load_data_from_aws(tables=args.tables, n_users=args.n_users)
    save_data(data, output_dir=args.output)

    print("\nDataFrames guardados:")
    for name, df in data.items():
        print(f"  {name:<25}: {df.shape[0]:>10,} filas × {df.shape[1]} cols")