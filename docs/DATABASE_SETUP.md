# Documentación: Puesta en Marcha de la Base de Datos - InstaCart RecSys

## Tabla de Contenidos
1. [Descripción General](#descripción-general)
2. [Requisitos Previos](#requisitos-previos)
3. [Configuración del Entorno](#configuración-del-entorno)
4. [Creación de la Base de Datos](#creación-de-la-base-de-datos)
5. [Ingestación de Datos](#ingestación-de-datos)
6. [Estructura de la Base de Datos](#estructura-de-la-base-de-datos)
7. [Verificación y Validación](#verificación-y-validación)
8. [Troubleshooting](#troubleshooting)
9. [Alternativas: Supabase (Cloud)](#alternativas-supabase-cloud)

---

## Descripción General

Este proyecto utiliza **PostgreSQL** como motor de base de datos principal para almacenar datos de InstaCart provenientes de Kaggle. El proceso de setup consta de dos etapas principales:

1. **Creación de Esquemas y Tablas**: Mediante el script SQL `InstaCart_DataBase_Creation.sql`
2. **Ingestación de Datos**: Mediante el script Python `data_ingestation.py`

### 🌩️ Nota sobre Supabase

Este proyecto también soporta **Supabase** como alternativa de PostgreSQL en la nube. Si prefieres:
- ✅ No instalar PostgreSQL localmente
- ✅ Acceso remoto a la BD
- ✅ Backups automáticos
- ✅ Servidor gestionado

**→ Consulta [SUPABASE_SETUP.md](SUPABASE_SETUP.md) para instrucciones completas**

El mismo script Python funciona con ambas opciones configurando la variable `DB_TYPE` en el archivo `.env`.

### Archivos Involucrados

| Archivo | Ubicación | Descripción |
|---------|-----------|-------------|
| `InstaCart_DataBase_Creation.sql` | `data/local_database/` | Script SQL para crear esquemas y tablas |
| `data_ingestation.py` | `src/data/` | Script Python para ingestación de datos desde CSVs |
| `SUPABASE_SETUP.md` | `docs/` | Guía para usar Supabase como alternativa (cloud) |

---

## Requisitos Previos

### 1. Software Requerido

- **PostgreSQL 12+**: Motor de base de datos
- **Python 3.8+**: Para ejecutar scripts de ingestación
- **pip**: Gestor de paquetes de Python

### 2. Instalación de PostgreSQL

#### En Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo service postgresql start
```

#### En macOS
```bash
brew install postgresql
brew services start postgresql
```

#### En Windows
Descargar e instalar desde: https://www.postgresql.org/download/windows/

### 3. Dependencias Python

Instalar las dependencias requeridas:

```bash
pip install -r requirements.txt
```

**Dependencias clave para el proyecto:**
- `psycopg2`: Adaptador PostgreSQL para Python
- `pandas`: Manipulación de datos
- `faker`: Generación de datos sintéticos
- `python-dotenv`: Gestión de variables de entorno
- `tqdm`: Barras de progreso

---

## Configuración del Entorno

### 1. Crear la Base de Datos en PostgreSQL

Acceder a PostgreSQL como superusuario:

```bash
sudo -u postgres psql
```

Crear la base de datos:

```sql
CREATE DATABASE instacart_database;
```

Salir de PostgreSQL:

```sql
\q
```

### 2. Configurar Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```bash
touch .env
```

#### Opción A: PostgreSQL Local

Agregar las siguientes variables:

```env
DB_TYPE=local
host=localhost
database=instacart_database
user=postgres
password=tu_contraseña_aqui
port=5432
```

#### Opción B: Supabase (Cloud)

Agregar las siguientes variables:

```env
DB_TYPE=supabase
SUPABASE_HOST=xxxxx.supabase.co
SUPABASE_DATABASE=postgres
SUPABASE_USER=postgres
SUPABASE_PASSWORD=tu_contraseña_supabase
SUPABASE_PORT=5432
```

**Para credenciales de Supabase:** Ver [SUPABASE_SETUP.md](SUPABASE_SETUP.md)

**⚠️ Importante:** 
- No subir el archivo `.env` a control de versiones
- Asegurarse de que `.env` está incluido en `.gitignore`

### 3. Verificar la Conexión

Para PostgreSQL local:

```bash
psql -h localhost -U postgres -d instacart_database
```

Para Supabase:

```bash
psql -h xxxxx.supabase.co -U postgres -d postgres
```

Si la conexión es exitosa, salir con `\q`.

---

## Creación de la Base de Datos

### Paso 1: Ejecutar el Script SQL

El script `InstaCart_DataBase_Creation.sql` crea todos los esquemas y tablas necesarios:

```bash
psql -h localhost -U postgres -d instacart_database -f data/local_database/InstaCart_DataBase_Creation.sql
```

**Alternativamente, desde dentro de PostgreSQL:**

```bash
psql -h localhost -U postgres -d instacart_database
```

Luego ejecutar:

```sql
\i data/local_database/InstaCart_DataBase_Creation.sql
```

### Paso 2: Verificar Esquemas y Tablas Creadas

Conectarse a la base de datos:

```bash
psql -h localhost -U postgres -d instacart_database
```

Listar esquemas:

```sql
\dn
```

Verificar tablas en cada esquema:

```sql
\dt users_schema.* orders_schema.* resources.* departments.*;
```

---

## Ingestación de Datos

### Requisitos: Archivos CSV de Kaggle

El script `data_ingestation.py` requiere los siguientes archivos CSV en la carpeta `data/`:

| Archivo CSV | Fuente | Descripción |
|-------------|--------|-------------|
| `departments.csv` | Kaggle InstaCart | Información de departamentos |
| `aisles.csv` | Kaggle InstaCart | Información de pasillos |
| `products.csv` | Kaggle InstaCart | Catálogo de productos |
| `orders.csv` | Kaggle InstaCart | Información de órdenes |
| `order_products_prior.csv` | Kaggle InstaCart | Productos en órdenes previas |
| `order_products_train.csv` | Kaggle InstaCart | Productos en órdenes de entrenamiento |

**Descargar datos:** https://www.kaggle.com/c/instacart-market-basket-analysis/data

### Paso 1: Descargar y Colocar CSVs

1. Descargar el dataset de Kaggle
2. Descomprimir los archivos
3. Colocar los CSVs en la carpeta: `data/`

```
data/
├── departments.csv
├── aisles.csv
├── products.csv
├── orders.csv
├── order_products_prior.csv
└── order_products_train.csv
```

### Paso 2: Ejecutar el Script de Ingestación

```bash
python src/data/data_ingestation.py
```

**Comportamiento del script:**

1. **Conexión a BD**: Valida conexión usando variables de `.env`
2. **Generación de Usuarios**: Crea 206,209 usuarios sintéticos con Faker
3. **Generación de Order IDs**: Crea 3,421,083 IDs de órdenes
4. **Inserción de Datos Sintéticos**: Inserta usuarios e IDs en tablas correspondientes
5. **Carga de CSVs**: Carga datos de Kaggle mediante COPY (muy rápido)
6. **Generación de Reporte**: Crea archivo `generation_summary.json`

**Tiempo estimado:** 5-15 minutos dependiendo del sistema

### Paso 3: Monitorear el Progreso

El script genera logs en tiempo real:
- **Consola**: Mensajes de progreso con barras de estado
- **Archivo**: `reports/logs/data_ingestation.log`

Ejemplo de salida:

```
2026-03-05 10:15:23 - INFO - Conexion exitosa
2026-03-05 10:15:24 - INFO - Generando 206209 usuarios
Generando Usuarios: 100%|██████████| 206209/206209
2026-03-05 10:18:33 - INFO - Usuarios insertados correctamente
...
2026-03-05 10:25:45 - INFO - RESUMEN DE GENERACIÓN DE DATOS
```

---

## Estructura de la Base de Datos

### Esquemas Creados

#### 1. **Users_Schema**
Almacena información de usuarios

```
users_schema.users
├── user_id (PK, SERIAL)
├── user_name (VARCHAR 50)
├── user_address (VARCHAR 500)
└── user_age (INT)
```

#### 2. **Resources**
Almacena información de recursos del catálogo

```
resources.aisles
├── aisle_id (PK, SERIAL)
└── aisle (VARCHAR 200)

resources.products
├── product_id (PK, SERIAL)
├── product_name (VARCHAR 500)
├── aisle_id (FK → aisles)
└── department_id (FK → departments)
```

#### 3. **Departments**
Almacena información de departamentos

```
departments.departments
├── department_id (PK, SERIAL)
└── department (VARCHAR 200)
```

#### 4. **Orders_Schema**
Almacena información de órdenes y sus productos

```
orders_schema.order
└── order_id (PK, SERIAL)

orders_schema.orders
├── order_id (INT, FK → order)
├── user_id (INT, FK → users)
├── eval_set (VARCHAR 6)
├── order_number (INT)
├── order_dow (INT)
├── order_hour_of_day (VARCHAR 2)
└── days_since_prior_order (FLOAT)

orders_schema.order_products_train
├── order_id (FK → order)
├── product_id (FK → products)
├── add_to_cart_order (INT)
└── reordered (INT)

orders_schema.order_products_prior
├── order_id (FK → order)
├── product_id (FK → products)
├── add_to_cart_order (INT)
└── reordered (INT)
```

### Diagrama Entidad-Relación

```
┌─────────────────────────────────────────────────────────────────┐
│                         Users_Schema                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ users (user_id, user_name, user_address, user_age)        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────┬──────────────────────────────────────┘
                          │ 1:N
                          │ (user_id)
                          ▼
┌────────────────────────────────────────────────────────────────┐
│                      Orders_Schema                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ orders (order_id, user_id, eval_set, ...)               │ │
│  └──────────────────────────────────────────────────────────┘ │
│                      │                                          │
│                      │ N:N                                      │
│    ┌────────────────┼────────────────┐                         │
│    │                │                │                         │
│    ▼                ▼                ▼                          │
│  ┌──────────┐    ┌────────┐    ┌────────────┐               │
│  │  order   │    │products│    │order_xxx   │               │
│  └──────────┘    └────────┘    └────────────┘               │
└────────────────────────────────────────────────────────────────┘
                          
┌────────────────────────────────────────────────────────────────┐
│                        Resources                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ products (product_id, product_name, aisle_id, dept_id)   │ │
│  └──────────────────────────────────────────────────────────┘ │
│      │ N:1                        │ N:1                        │
│      ▼                            ▼                            │
│  ┌──────────┐               ┌──────────────┐               │
│  │ aisles   │               │departments   │               │
│  └──────────┘               └──────────────┘               │
└────────────────────────────────────────────────────────────────┘
```

---

## Verificación y Validación

### Consultas de Verificación

Conectarse a la base de datos:

```bash
psql -h localhost -U postgres -d instacart_database
```

#### 1. Contar registros por tabla

```sql
-- Usuarios
SELECT COUNT(*) as total_usuarios FROM users_schema.users;

-- Órdenes
SELECT COUNT(*) as total_ordenes FROM orders_schema.order;

-- Productos
SELECT COUNT(*) as total_productos FROM resources.products;

-- Departamentos
SELECT COUNT(*) as total_departamentos FROM departments.departments;

-- Pasillos
SELECT COUNT(*) as total_aisles FROM resources.aisles;
```

#### 2. Verificar Integridad Referencial

```sql
-- Verificar que todos los user_ids en orders existen en users
SELECT COUNT(*) FROM orders_schema.orders o 
WHERE NOT EXISTS (SELECT 1 FROM users_schema.users u WHERE u.user_id = o.user_id);
-- Debería retornar 0

-- Verificar que todos los product_ids en order_products_train existen
SELECT COUNT(*) FROM orders_schema.order_products_train opt 
WHERE NOT EXISTS (SELECT 1 FROM resources.products p WHERE p.product_id = opt.product_id);
-- Debería retornar 0
```

#### 3. Inspeccionar Datos de Muestra

```sql
-- Primeros 5 usuarios
SELECT * FROM users_schema.users LIMIT 5;

-- Productos con department y aisle info
SELECT p.product_name, d.department, a.aisle
FROM resources.products p
JOIN departments.departments d ON p.department_id = d.department_id
JOIN resources.aisles a ON p.aisle_id = a.aisle_id
LIMIT 10;
```

### Archivo de Reporte Generado

Después de ejecutar `data_ingestation.py`, se crea `generation_summary.json`:

```json
{
  "generation_date": "2026-03-05T10:28:45.123456",
  "total_records": 13500000,
  "table_counts": {
    "departments": 21,
    "order": 3421083,
    "order_products_prior": 3300000,
    "order_products_train": 1500000,
    "orders": 3421083,
    "aisles": 134,
    "products": 49688,
    "users": 206209
  }
}
```

---

## Troubleshooting

### Problema: Error de Conexión a PostgreSQL

**Síntoma:**
```
psycopg2.OperationalError: could not connect to server
```

**Soluciones:**

1. Verificar que PostgreSQL está corriendo:
```bash
sudo service postgresql status
```

2. Si no está activo, iniciar:
```bash
sudo service postgresql start
```

3. Verificar credenciales en `.env` (host, user, password, port)

4. Verificar conectividad a PostgreSQL:
```bash
psql -h localhost -U postgres
```

---

### Problema: Archivo .env No Encontrado

**Síntoma:**
```
Error: Invalid PostgreSQL connection parameters
```

**Solución:**
```bash
# Crear archivo .env en la raíz del proyecto
touch .env

# Agregar variables de conexión
echo "host=localhost" >> .env
echo "database=instacart_database" >> .env
echo "user=postgres" >> .env
echo "password=tu_contraseña" >> .env
echo "port=5432" >> .env
```

---

### Problema: Archivos CSV No Encontrados

**Síntoma:**
```
Error al conectar: El archivo ./data/departments.csv no existe. Omitir...
```

**Solución:**

1. Verificar que los CSVs están en la carpeta `data/`:
```bash
ls -la data/
```

2. Asegurar que los nombres de archivo son exactos:
   - `departments.csv`
   - `aisles.csv`
   - `products.csv`
   - `orders.csv`
   - `order_products_prior.csv`
   - `order_products_train.csv`

3. Descargar de Kaggle si están faltando

---

### Problema: Restricción de Clave Foránea Violada

**Síntoma:**
```
violates foreign key constraint
```

**Causa Probable:** Datos inconsistentes o script SQL incompleto

**Soluciones:**

1. Eliminar y recrear la base de datos:
```bash
# Conectarse como postgres
psql -U postgres

# Eliminar BD
DROP DATABASE IF EXISTS instacart_database;

# Crear BD nueva
CREATE DATABASE instacart_database;

# Ejecutar script SQL nuevamente
\i data/local_database/InstaCart_DataBase_Creation.sql
```

2. Verificar que el script SQL está completo y sin errores

---

### Problema: Proceso de Ingestación Lento

**Síntoma:** Ingestación toma más de 30 minutos

**Optimizaciones:**

1. Usar SSD en lugar de HDD

2. Aumentar `shared_buffers` en PostgreSQL:
```sql
-- Como superusuario
ALTER SYSTEM SET shared_buffers = '256MB';
-- Reiniciar PostgreSQL
```

3. Desactivar synchronous_commit temporalmente:
```sql
ALTER DATABASE instacart_database SET synchronous_commit = OFF;
```

4. Crear índices después de la carga (no durante):
```sql
-- Ejecutar después de data_ingestation.py
CREATE INDEX idx_order_user_id ON orders_schema.orders(user_id);
CREATE INDEX idx_product_aisle ON resources.products(aisle_id);
CREATE INDEX idx_product_dept ON resources.products(department_id);
```

---

### Problema: Permiso Denegado

**Síntoma:**
```
permission denied for schema public
```

**Solución:**

Otorgar permisos al usuario:

```sql
-- Como superusuario (postgres)
GRANT ALL PRIVILEGES ON DATABASE instacart_database TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA users_schema TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA orders_schema TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA resources TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA departments TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA users_schema TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA orders_schema TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA resources TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA departments TO postgres;
```

---

## Proceso Completo: Checklist

Para ejecutar todo el setup desde cero:

- [ ] 1. PostgreSQL instalado y corriendo
- [ ] 2. Python 3.8+ instalado
- [ ] 3. Dependencias instaladas: `pip install -r requirements.txt`
- [ ] 4. Base de datos creada: `CREATE DATABASE instacart_database;`
- [ ] 5. Archivo `.env` configurado correctamente
- [ ] 6. Archivo de configuración a nivel sistema (conexión verificada)
- [ ] 7. Script SQL ejecutado: `InstaCart_DataBase_Creation.sql`
- [ ] 8. CSVs descargados de Kaggle y colocados en `data/`
- [ ] 9. Script Python ejecutado: `python src/data/data_ingestation.py`
- [ ] 10. Verificación de conteos y reporte generado
- [ ] 11. Consultas de integridad validadas

---

## Recursos Adicionales

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Kaggle InstaCart Dataset](https://www.kaggle.com/c/instacart-market-basket-analysis)
- [psycopg2 Documentation](https://www.psycopg.org/2/)
- [Python Faker Library](https://faker.readthedocs.io/)

---

## Contacto y Soporte

Para problemas o sugerencias sobre este setup, consultar con el equipo de desarrollo del proyecto.

**Última actualización:** Marzo 5, 2026
