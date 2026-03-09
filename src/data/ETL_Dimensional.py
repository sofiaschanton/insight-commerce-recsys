import psycopg2
import logging
import os
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/logs/dimensional_etl.log'),
        logging.StreamHandler()
    ]
)

load_dotenv(dotenv_path='/home/asus_juan/Documents/GitHub/insight-commerce-recsys/.env')

# Credenciales Locales (Importante ajustar en el .env con los mismo nombres de variable)
LOCAL_DB_CONFIG = {
    'host': os.getenv('host'),
    'database': os.getenv('database'),
    'user': os.getenv('user'),
    'password': os.getenv('password'),
    'port': os.getenv('port')
}

# Credenciales Supabase (Para saber donde encontrar las credeciales revisa /docs/SUPABASE_SETUP.md)
SUPA_DB_CONFIG = {
    'host': os.getenv('host_sup'),
    'database': os.getenv('database_sup'),
    'user': os.getenv('user_sup'),
    'password': os.getenv('password_sup'),
    'port': os.getenv('port_sup')
}

class DimensionalETL:
    def __init__(self, local_config:dict, supa_config:dict):
        """
        Inicializa la clase DimensionalETL con las configuraciones de conexión.

        Args:
            local_config (dict): Diccionario con las credenciales de la base de datos PostgreSQL local.
            supa_config (dict): Diccionario con las credenciales de la base de datos PostgreSQL en Supabase.
        """

        self.local_config = local_config
        self.supa_config = supa_config
        self.local_conn = None
        self.supa_conn = None
        self.local_cursor = None
        self.supa_cursor = None
        self.batch_id = int(datetime.now().timestamp())

        self.report_stats = {}
        self.pipeline_start_time = None

    def connect(self):
        """
        Establece las conexiones a las bases de datos origen (Local) y destino (Supabase).

        Returns:
            bool: True si ambas conexiones se establecen correctamente, False en caso de error.
        """

        try:
            self.local_conn = psycopg2.connect(**self.local_config) # Conexión a la base de datos local
            self.supa_conn = psycopg2.connect(**self.supa_config)   # Conexión a la base de datos Supabase 
            logging.info("Conexiones a Local y Supabase exitosas.")
            return True
        except Exception as e:
            logging.error(f"Error al conectar a las bases de datos: {e}")
            return False

    def close(self):
        """
        Cierra de forma segura los cursores y las conexiones activas a ambas bases de datos.
        """

        if self.local_conn: self.local_conn.close()
        if self.supa_conn: self.supa_conn.close()
        logging.info("Conexiones cerradas.")

    # Debido a problemas con la carga y descarga de grandes volúmenes de datos, 
    # implementamos un método que utiliza cursores del lado del servidor para extraer datos en chunks 
    # y luego insertarlos en Supabase por lotes.
    # Esto ayuda a cuidar y proteger la RAM de la máquina local y a evitar timeouts en la conexión con Supabase.

    def transfer_data(self, extract_query:str, target_table:str, insert_query:str, chunk_size:int=10000):
        """
        Extrae datos de la base de datos local utilizando un cursor del lado del servidor 
        y los inserta en Supabase por lotes para optimizar el uso de memoria RAM.

        Calcula la duración de la transferencia y registra las métricas de éxito o error 
        en el diccionario interno report_stats para la generación del reporte final.

        Args:
            extract_query (str): Consulta SQL SELECT para extraer los datos del origen.
            target_table (str): Nombre de la tabla de destino en Supabase.
            insert_query (str): Consulta SQL INSERT parametrizada (con %s) para el destino.
            chunk_size (int, optional): Cantidad de registros a extraer e insertar por lote. 
                                        Por defecto es 10000.
        """

        logging.info(f"Iniciando transferencia para la tabla: {target_table}")
        
        # Esto se usara en el reporte
        start_time = datetime.now()
        total_inserted = 0
        status = "Completado"
        error_msg = "Ninguno"
        
        # Intetamos hacer la conexione entre las bases de datos y creamos el cursor
        try:
            local_cursor = self.local_conn.cursor(name=f'fetch_cursor_{target_table}')
            local_cursor.execute(extract_query) # Ejecutamos la query de extraccion de la base de datos local
            supa_cursor = self.supa_conn.cursor() # Conectamos a el cursor de supabase para insertar los datos
            
            while True:
                # Fetchmany nos permite traer un chunk de registros a la vez 
                # lo que es mucho más eficiente para grandes volúmenes de datos.
                records = local_cursor.fetchmany(chunk_size) 
                if not records:
                    break
                
                # Mediante execute_values podemos insertar un lote de registros en Supabase de manera eficiente.
                # Esto con la finalidad de reducir la cantidad de round-trips entre la aplicación y la base de datos
                # lo que mejora el rendimiento.
                execute_values(supa_cursor, insert_query, records, page_size=chunk_size)
                self.supa_conn.commit()
                
                total_inserted += len(records)
                logging.info(f"{target_table}: {total_inserted} registros transferidos...")

            local_cursor.close()
            supa_cursor.close()
            logging.info(f"Transferencia de {target_table} completada. Total: {total_inserted}")

        except Exception as e:
            self.supa_conn.rollback()
            status = "Fallido"
            error_msg = str(e)
            logging.error(f"Error transfiriendo {target_table}: {e}")
            
        # Para el reporte en logs
        finally:
            duration = datetime.now() - start_time # Lo que se demoro en hacer el proceso de transferencia de datos
            # Este diccionario guarda el estado, la cantidad de filas procesadas, la duracion y si hay algun error
            self.report_stats[target_table] = {
                "estado": status,
                "filas_procesadas": total_inserted,
                "duracion": duration,
                "error": error_msg
            }

    def generate_report(self):
        """
        Genera y emite un reporte estructurado en los logs del sistema.

        Recopila las métricas almacenadas en report_stats (filas procesadas, duración, estados)
        con una salida en texto que resume el rendimiento general del ETL y los detalles por cada tabla procesada.
        """

        if not self.pipeline_start_time:
            logging.warning("El pipeline no ha sido iniciado correctamente para generar un reporte.")
            return

        total_duration = datetime.now() - self.pipeline_start_time
        total_rows = sum(stats['filas_procesadas'] for stats in self.report_stats.values())
        
        # Visual del reporte
        report_lines = [
            "",
            "="*75,
            "  REPORTE DE EJECUCION - ETL Dimensional - Instacart Dataset    ",
            "="*75,
            f"Batch ID        : {self.batch_id}",
            f"Hora de Inicio  : {self.pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Tiempo Total    : {total_duration}",
            f"Total Filas     : {total_rows:,}",
            "-"*75,
            f"{'TABLA DESTINO':<22} | {'ESTADO':<12} | {'FILAS':<10} | {'DURACIÓN':<15}",
            "-"*75
        ]

        # Creamos el reporte por cada table procesada
        for table, stats in self.report_stats.items():
            # Convertimos la duración a un formato legible (hh:mm:ss) y lo agregamos al reporte
            dur_str = str(stats['duracion']).split('.')[0] 
            report_lines.append(
                f"{table:<22} | {stats['estado']:<12} | {stats['filas_procesadas']:<10,} | {dur_str:<15}"
            )
            
            # Si hubo un error en esta tabla, lo agregamos
            if stats['estado'] == 'Fallido':
                report_lines.append(f"  -> ERROR: {stats['error']}")

        report_lines.append("="*75)
        report_lines.append("")

        # Imprimimos el reporte completo de una sola vez
        logging.info("\n".join(report_lines))

    def run_pipeline(self):
        """
        Orquesta la ejecución completa del pipeline ETL.

        Sigue el flujo de:
        1. Registrar el tiempo de inicio global.
        2. Conectar a las bases de datos.
        3. Ejecutar las transferencias secuenciales (dim_user, dim_product, fact_order_products)
        aplicando las lógicas de negocio expresadas en el EDA y desnormalización requeridas.
        4. Cerrar las conexiones.
        5. Generar el reporte de la ejecución.
        """

        self.pipeline_start_time = datetime.now()

        if not self.connect():
            return

        # 1. DIM_USER
        # Transformación: Calculamos el año de nacimiento aproximado, para poder tener una dimension 
        # de edad que no dependa de la fecha de la captura de los datos si no de la edad real
        query_ext_user = """
            SELECT 
                user_id AS user_key,
                user_name,
                user_address,
                DATE_TRUNC('year', CURRENT_DATE - MAKE_INTERVAL(years := user_age))::DATE AS user_birthdate
            FROM Users_Schema.users;
        """
        query_ins_user = """
            INSERT INTO dim_user (user_key, user_name, user_address, user_birthdate) 
            VALUES %s ON CONFLICT (user_key) DO NOTHING;
        """
        self.transfer_data(query_ext_user, 'dim_user', query_ins_user)

        # 2. DIM_PRODUCT
        # Transformación: Desnormalizamos uniendo productos, aisles y departamentos.
        query_ext_product = """
            SELECT 
                p.product_id AS product_key,
                p.product_name,
                a.aisle AS aisle_name,
                d.department AS department_name
            FROM Resources.products p
            JOIN Resources.aisles a ON p.aisle_id = a.aisle_id
            JOIN Departments.departments d ON p.department_id = d.department_id;
        """
        query_ins_product = """
            INSERT INTO dim_product (product_key, product_name, aisle_name, department_name) 
            VALUES %s ON CONFLICT (product_key) DO NOTHING;
        """
        self.transfer_data(query_ext_product, 'dim_product', query_ins_product)

        # 3. FACT_ORDER_PRODUCTS
        # Transformación: Unimos las órdenes con los detalles (prior y train) 
        # para traer el target 'reordered' y el 'eval_set'.
        query_ext_fact = """
            WITH All_Orders AS (
                -- 1. Unificamos las ordenes de Prior y Train para no repetir subconsultas
                SELECT order_id, product_id, add_to_cart_order, reordered 
                FROM Orders_Schema.order_products_prior
                UNION ALL
                SELECT order_id, product_id, add_to_cart_order, reordered 
                FROM Orders_Schema.order_products_train
            ),
            Clientes_Frecuentes AS (
                -- 2. Condición #1: Clientes con más de 5 órdenes en total
                -- FIX: Se eliminó ORDER BY y LIMIT que sesgaban la selección de usuarios
                SELECT user_id
                FROM Orders_Schema.orders
                GROUP BY user_id
                HAVING COUNT(order_id) > 5
                LIMIT 20000
            ),
            Productos_Reordenados AS (
                -- 3. Condición #2: Productos que se han reordenado más de 50 veces (globalmente)
                SELECT product_id
                FROM All_Orders 
                WHERE reordered = 1
                GROUP BY product_id
                HAVING COUNT(*) > 50
            ),
            Ordenes_Con_Populares AS (
                -- 4. Condición #3: Todas las ordenes que almenos tengan un producto reordenado mas de 50 veces
                SELECT DISTINCT order_id
                FROM All_Orders
                WHERE product_id IN (SELECT product_id FROM Productos_Reordenados)
            )
            -- Consulta Principal
            SELECT 
                o.order_id AS order_key,
                o.user_id AS user_key,
                op.product_id AS product_key,
                o.order_dow,
                CAST(o.order_hour_of_day AS SMALLINT),
                o.days_since_prior_order,
                op.add_to_cart_order,
                op.reordered,
                o.order_number,
                o.eval_set AS get_eval
            FROM Orders_Schema.orders o
            JOIN All_Orders op ON o.order_id = op.order_id
            -- El INNER JOIN actúa como filtro: si el usuario o la orden no existen en los CTEs, se descartan
            JOIN Clientes_Frecuentes cf ON o.user_id = cf.user_id
            JOIN Ordenes_Con_Populares ocp ON o.order_id = ocp.order_id
            WHERE o.eval_set IN ('prior', 'train');
        """

        query_ins_fact = """
            INSERT INTO fact_order_products (
                order_key, user_key, product_key, order_dow, order_hour_of_day, 
                days_since_prior_order, add_to_cart_order, reordered, order_number, get_eval
            ) VALUES %s ON CONFLICT (order_key, product_key) DO NOTHING;
        """
        
        # Usamos un chunk_size de 10,000 para balancear uso de RAM y velocidad de red hacia Supabase
        self.transfer_data(query_ext_fact, 'fact_order_products', query_ins_fact, chunk_size=10000)

        self.generate_report()
        
        self.close()

if __name__ == "__main__":
    etl = DimensionalETL(LOCAL_DB_CONFIG, SUPA_DB_CONFIG)
    etl.run_pipeline()