import psycopg2
import pandas as pd
import numpy as np
import random
import logging
import json
import os
from pathlib import Path
from psycopg2.extras import execute_batch
from tqdm import tqdm
from faker import Faker
from dotenv import load_dotenv
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[2]
log_path = ROOT_DIR / "reports" / "logs" / "data_loader.log"

# Configuracion inicial para guardar los logs para asi ver en que nos equivocamos y guardar un reporte
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_path)), # Path donde se guarda el archivo con los logs
        logging.StreamHandler()
    ]
)

load_dotenv(ROOT_DIR / '.env')

# Las buscamos en nuestro .env
local_host = os.getenv('local_host')
local_database = os.getenv('local_database') 
local_user = os.getenv('local_user')
local_password = os.getenv('local_password')
local_port = os.getenv('local_port')

# Configruacion de Postgres para hacer la conexion
DB_CONFIG = {
    'host': local_host,
    'database': local_database,
    'user': local_user,
    'password': local_password,
    'port': local_port
}

# Faker es una libreria que crea datos sinteticos, seran necesarios para algunos nombres
fake = Faker('en_US') # Ingles de USA
Faker.seed(42) 
random.seed(42)
np.random.seed(42)

# Creamos la clase de ingestacion de los datos a postgres
class DataIngestation:
    # Configuracion del __init__
    def __init__(self, db_config):
        """
        Inicializa la clase DataIngestation con la configuración de la base de datos.

        Args:
            db_config (dict): Diccionario con parámetros de conexión a PostgreSQL.
        """
        self.db_config = db_config
        self.connection = None
        self.cursor = None

        # Contador de datos
        self.counters = {
            'departments':0,
            'order':0,
            'order_products_prior':0,
            'order_products_train':0,
            'orders':0,
            'aisles':0,
            'products':0,
            'users':0
        }
    
    # Nos conectamos a la base de datos postgreSQL
    def connect(self):
        """
        Establece la conexión con la base de datos PostgreSQL usando la configuración proporcionada.

        Returns:
            bool: True si la conexión fue exitosa, False en caso contrario.
        """
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logging.info("Conexion exitosa")
            return True
        except Exception as e:
            logging.error(f"Error al conectar {e}")
            return False

    # Definimos cuando cerrar la conexion (Buena Practica)    
    def close(self):
        """
        Cierra el cursor y la conexión a la base de datos de forma segura.
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logging.info("Conexion cerrada")

    def table_has_records(self, schema, table):
        """
        Verifica si una tabla ya contiene registros.

        Args:
            schema (str): Esquema de la tabla.
            table (str): Nombre de la tabla.

        Returns:
            bool: True si la tabla tiene al menos un registro.
        """
        self.cursor.execute(f"SELECT EXISTS (SELECT 1 FROM {schema}.{table} LIMIT 1)")
        return self.cursor.fetchone()[0]

    def get_table_count(self, schema, table):
        """
        Obtiene la cantidad de registros actuales de una tabla.

        Args:
            schema (str): Esquema de la tabla.
            table (str): Nombre de la tabla.

        Returns:
            int: Cantidad de registros en la tabla.
        """
        self.cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        return self.cursor.fetchone()[0]
    
    # Generamos los usuarios
    def generate_users(self, count=206209):
        """
        Genera datos sintéticos de usuarios utilizando la librería Faker.

        Args:
            count (int, optional): Cantidad de usuarios a generar. Por defecto es 206209.

        Returns:
            list: Lista de tuplas con la información de los usuarios 
                  (user_id, user_name, user_address, user_age).
        """
        logging.info(f"Generando {count} usuarios")

        users = []

        for i in tqdm(range(1, count + 1), desc="Generando Usuarios"):

            user_id = i
            user_name = fake.first_name()
            user_address = fake.address().replace('\n', ',')
            user_age = random.randint(16, 80)

            users.append((
                user_id,
                user_name,
                user_address,
                user_age
            ))

        return users
    
    # Generamos los ID de ordenes 
    def generate_order(self, count=3421083):
        """
        Genera una lista de identificadores únicos para las Orders.

        Args:
            count (int, optional): Cantidad de órdenes a generar. Por defecto es 3421083. Numero maximo del dataset

        Returns:
            list: Lista con los IDs de las Orders generadas.
        """
        logging.info("Generando order_id")

        orders = []

        for i in tqdm(range(1, count + 1), desc="Generando Orders"):
            order_id = i
            orders.append((order_id,))

        return orders

    # Insertamos los usuarios a la tabla de usaurios mediante una query
    def insert_users(self, users_data, schema='users_schema', table='users'):
        """
        Inserta de forma masiva (batch) los datos de usuarios en la tabla correspondiente.

        Args:
            users_data (list): Lista de tuplas con los datos de los usuarios.
            schema (str, optional): Nombre del esquema en la base de datos. Por defecto 'users_schema'.
            table (str, optional): Nombre de la tabla. Por defecto 'users'.
        """
        logging.info(f"Insertando {len(users_data)} usuarios a {schema}.{table}..")

        insert_query = f"""
            INSERT INTO {schema}.{table} (
                user_id,
                user_name,
                user_address,
                user_age
            ) VALUES (%s, %s, %s, %s)
        """

        try:
            execute_batch(self.cursor, insert_query, users_data, page_size=10000)
            self.connection.commit()
            self.counters['users'] = len(users_data)
            logging.info("Usuarios insertados correctamente")
        except Exception as e:
            self.connection.rollback()
            logging.error(f"Error insertando usuarios: {e}")

    # Insertamos las ordenes id
    def insert_orders(self, orders_data, schema='orders_schema', table='order'):
        """
        Inserta de forma masiva (batch) los identificadores de las órdenes en la base de datos.

        Args:
            orders_data (list): Lista con los identificadores de las órdenes a insertar.
            schema (str, optional): Nombre del esquema en la base de datos. Por defecto 'orders_schema'.
            table (str, optional): Nombre de la tabla. Por defecto 'order'.
        """
        logging.info(f"Insertando {len(orders_data)} usuarios a {schema}.{table}..")

        insert_query = f"""
            INSERT INTO {schema}.{table} (
                order_id
            ) VALUES (%s)
        """

        try:
            execute_batch(self.cursor, insert_query, orders_data, page_size=10000)
            self.connection.commit()
            self.counters['order'] = len(orders_data)
            logging.info("Usuarios insertados correctamente")
        except Exception as e:
            self.connection.rollback()
            logging.error(f"Error insertando usuarios: {e}")

    # Esta funcion es la mas importante ya que esta permite cargar los datos desde los csv a las tablas
    def load_csv_kaggle_data(self, file_path, schema, table):
        """
        Carga datos desde un archivo CSV a una tabla en PostgreSQL usando el comando COPY.

        Args:
            file_path (str): Ruta completa al archivo CSV.
            schema (str): Nombre del esquema de la base de datos destino.
            table (str): Nombre de la tabla destino.
        """

        logging.info(f"Cargando datos desde {file_path} a {schema}.{table} usando COPY...")
        if not os.path.exists(file_path):
            logging.error(f"El archivo {file_path} no existe. Omitir...")
            return

        try:
            with open(file_path, 'r', encoding='utf-8')as f:
                copy_query = f"COPY {schema}.{table} FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER ',')" # Query SQL para copiar un CSV en una tabla
                self.cursor.copy_expert(copy_query, f)
            
            self.connection.commit()
            logging.info(f"Carga completa para {schema}.{table}")

            self.cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
            self.counters[table] = self.cursor.fetchone()[0]
        except Exception as e:
            self.connection.rollback()
            logging.error(f"Error cargando {file_path}: {e}")
    
    def generate_summary_report(self):
        """
        Genera un reporte de resumen con los conteos totales de registros insertados en las tablas
        y guarda este reporte en un archivo JSON local.
        """
        logging.info("\n RESUMEN DE GENERACIÓN DE DATOS")
        logging.info("="*50)
        
        # Conteos finales
        tables = [            
            'departments.departments',
            'orders_schema.order',
            'orders_schema.order_products_prior',
            'orders_schema.order_products_train',
            'orders_schema.orders',
            'resources.aisles',
            'resources.products',
            'users_schema.users'
        ]
        total_records = 0
        
        for table in tables:
            self.cursor.execute(f"select count(*) from {table}")
            count = self.cursor.fetchone()[0]
            logging.info(f"  {table}: {count:,} registros")
            total_records += count
        
        logging.info(f"\n  TOTAL: {total_records:,} registros")
        
        # Guardar resumen en JSON
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_records': total_records,
            'table_counts': self.counters,
        }
        
        with open('generation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info("\n Resumen guardado en generation_summary.json")   
    
    def run_etl(self, csv_directory):
        """
        Ejecuta el proceso general de ETL: crea datos sintéticos de usuarios y órdenes,
        los inserta en la base de datos y luego carga los archivos CSV indicados.

        Args:
            csv_directory (str): Ruta al directorio que contiene los archivos CSV.
        """
        if not self.connect():
            return

        if self.table_has_records('users_schema', 'users'):
            current_users = self.get_table_count('users_schema', 'users')
            self.counters['users'] = current_users
            logging.info(f"users_schema.users ya contiene datos ({current_users}). Se omite la carga de usuarios")
        else:
            users_data = self.generate_users()
            self.insert_users(users_data, schema='users_schema', table='users')

        if self.table_has_records('orders_schema', 'order'):
            current_orders = self.get_table_count('orders_schema', 'order')
            self.counters['order'] = current_orders
            logging.info(f"orders_schema.order ya contiene datos ({current_orders}). Se omite la carga de order_id")
        else:
            orders_data = self.generate_order()
            self.insert_orders(orders_data, schema='orders_schema', table='order')

        csv_base_path = Path(csv_directory)
        if not csv_base_path.is_absolute():
            csv_base_path = (ROOT_DIR / csv_base_path).resolve()

        if not csv_base_path.exists():
            logging.error(f"El directorio de CSV no existe: {csv_base_path}")
            self.close()
            return

        files_mapping = [
        # Primero las tablas de referencia (sin dependencias)
        ('departments.csv', 'departments', 'departments'),
        ('aisles.csv', 'resources', 'aisles'),
        ('products.csv', 'resources', 'products'),
        # Después las tablas que dependen de products y orders
        ('orders.csv', 'orders_schema', 'orders'),
        ('order_products__prior.csv', 'orders_schema', 'order_products_prior'),
        ('order_products__train.csv', 'orders_schema', 'order_products_train'),
        ]

        for file_name, schema, table in files_mapping:
            if self.table_has_records(schema, table):
                current_count = self.get_table_count(schema, table)
                self.counters[table] = current_count
                logging.info(f"{schema}.{table} ya contiene datos ({current_count}). Se omite la carga de {file_name}")
                continue

            file_path = str(csv_base_path / file_name)
            self.load_csv_kaggle_data(file_path, schema, table)

        logging.info("Resumen")
        for table, count in self.counters.items():
            logging.info(f"{table}: {count} registros")

        self.close()

if __name__ == "__main__":

    PATH_TO_CSVS = "data/raw" # Colocar la ruta de los CSV

    etl = DataIngestation(DB_CONFIG)
    etl.run_etl(PATH_TO_CSVS)