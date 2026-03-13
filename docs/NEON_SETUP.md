# Documentación: Configuración de Neon - Insight Commerce

## Tabla de Contenidos
1. [¿Qué es Neon?](#qué-es-neon)
2. [Ventajas de Usar Neon](#ventajas-de-usar-neon)
3. [Setup en Neon](#setup-en-neon)
4. [Obtener Credenciales](#obtener-credenciales)
5. [Configuración del `.env`](#configuración-del-env)
6. [Connection Pooling en Neon](#connection-pooling-en-neon)
7. [Ejecución con Neon](#ejecución-con-neon)
8. [Dashboard de Neon](#dashboard-de-neon)
9. [Troubleshooting](#troubleshooting)

---

## ¿Qué es Neon?

**Neon** es una plataforma PostgreSQL Serverless construida desde cero para la nube. A diferencia de las bases de datos tradicionales, Neon:

- 🚀 **Separa almacenamiento de cómputo**: El almacenamiento y el computo son independientes, permitiendo escalado eficiente.
- 🔄 **Auto-suspend (Escalado a Cero)**: El computo se suspende automáticamente tras 5 minutos de inactividad (en plan free), ahorrando costos.
- 🌿 **Database Branching**: Crear ramas de BD similar a Git (ideal para desarrollo y pruebas sin afectar producción).
- 🔐 **PostgreSQL Completo**: Compatible 100% con PostgreSQL, sin cambios en tu código.
- 📊 **Connection Pooling Nativo**: PgBouncer integrado para gestionar miles de conexiones sin problemas.
- ⚡ **Escalado Automático**: Aumenta o disminuye recursos automáticamente según demanda.

Para este proyecto usamos **Neon como PostgreSQL remoto serverless**, aprovechando su bajo costo en plan gratuito y flexibilidad para escalar.

---

## Ventajas de Usar Neon Vs. Alternativas

| Característica | Local | Supabase | AWS RDS | Neon |
|----------------|-------|----------|---------|------|
| **Instalación** | Compleja (PostgreSQL) | 2 minutos | Complicada (AWS) | 2 minutos |
| **Mantenimiento** | Responsabilidad tuya | Gestionado | AWS lo gestiona | Totalmente gestionado |
| **Auto-suspend** | No | No | No | Sí ✅ |
| **Database Branching** | No | No | No | Sí ✅ |
| **Connection Pooling** | Manual (pgBouncer) | No | Manual | Nativo con PgBouncer ✅ |
| **Pricing (Free Tier)** | $0 (tu hardware) | $0 hasta 500MB | No hay free tier | $0 + créditos ✅ |
| **Escalado Automático** | No | No | Manual | Automático ✅ |
| **Cold Start** | N/A | N/A | N/A | 1-2 seg tras 5 min ⏱️ |

### Plan Gratuito de Neon

```
Proyecto Gratis (Ideal para desarrollo y producción inicial):
├─ Almacenamiento: 500 MB
├─ Compute (vCPU): Compartido, auto-suspend tras 5 min inactividad
├─ Conexiones: 
│  ├─ Directas: 100 por endpoint
│  └─ Pooled: Ilimitadas (via PgBouncer)
├─ Proyectos: 1 proyecto
├─ Ramas: 1 rama (main)
├─ Backups: Diarios
├─ Cold Storage: Soportado
└─ Período de prueba: 30 días con $300 en créditos
```

---

## Setup en Neon

### Paso 1: Crear Cuenta

1. Ir a [neon.tech](https://neon.tech)
2. Hacer clic en **"Sign Up"** (arriba a la derecha)
3. Crear cuenta con Google, GitHub o email
4. Verificar email

### Paso 2: Crear Proyecto (Branch)

1. Una vez autenticado, verás tu **Dashboard**
2. Hacer clic en **"Create Project"**
3. Completar formulario:
   - **Project name**: `instacart-recsys`
   - **Database name**: `instacart_database` (⚠️ Importante: mantener el mismo nombre)
   - **Region**: Seleccionar región más cercana (ej: `US East (Ohio)`)
   - **Postgres version**: Última versión estable (15.x o superior)
4. Esperar 30-60 segundos a que se cree

**Nota importante**: Neon toma nota de tu contraseña inicial (no la puedes cambiar aquí, pero sí en Project Settings).

### Paso 3: Obtener Connection String

1. En el Dashboard, ver tu proyecto recién creado
2. Hacer clic en **"Connect"** (botón azul arriba a la derecha)
3. Copiar el **Connection string** (URL completa como: `postgresql://user:password@host/database`)

### Paso 4: Crear Esquemas y Tablas

1. En el panel izquierdo, hacer clic en **"SQL Editor"**
2. Hacer clic en **"New Query"**
3. Copiar y pegar el contenido completo de `data/local_database/InstaCart_DataBase_Creation.sql`
4. Ejecutar con botón **"Execute"** (o Ctrl+Enter)

**Tiempo estimado**: 10-30 segundos (depende del tamaño del script)

**Nota**: Las tablas se crearán dentro del esquema especificado en el SQL (usualmente `users_schema`, `resources`, etc.)

---

## Obtener Credenciales

### Desde el Connection String (Recomendado)

Neon proporciona la conexión en formato URI. Descomponerla así:

```
postgresql://[NEON_USER]:[NEON_PASSWORD]@[NEON_HOST]/[NEON_DATABASE]
```

**Ejemplo**:
```
postgresql://neonuser:abcdef123456@ep-cool-breeze-123456.us-east-1.neon.tech/instacart_database
```

Se descompone en:
```
NEON_USER=neonuser
NEON_PASSWORD=abcdef123456
NEON_HOST=ep-cool-breeze-123456.us-east-1.neon.tech
NEON_PORT=5432 (por defecto, no especificar)
NEON_DATABASE=instacart_database
```

### Desde el Dashboard

1. Ir a **Project Settings** (engranaje arriba a la derecha)
2. Ir a **Connection** → **Connection String**
3. Ver todos los parámetros individuales

### Credenciales Especiales en Neon

**Important**: Neon ofrece dos tipos de conexión:

| Tipo | Host | Uso |
|------|------|-----|
| **Direct** | `ep-xxxxx.us-east-1.neon.tech` | Scripts, migraciones, data ingestación |
| **Pooled** | `ep-xxxxx-pooler.us-east-1.neon.tech` | APIs, aplicaciones web, conexiones persistentes |

Para este proyecto usaremos ambas (explicado más adelante).

---

## Configuración del `.env`

Crear o editar el archivo `.env` en la raíz del proyecto:

```bash
# ===== OPCIÓN 1: Usar Neon (Serverless Postgres) =====
DB_TYPE=neon

# Conexión DIRECTA (para data_ingestation.py y scripts pesados)
NEON_HOST=ep-xxxxx.us-east-1.neon.tech
NEON_DATABASE=instacart_database
NEON_USER=neonuser
NEON_PASSWORD=tu_contrasena_super_segura
NEON_PORT=5432

# Conexión POOLED (para APIs y aplicaciones web - opcional por ahora)
NEON_POOLER_HOST=ep-xxxxx-pooler.us-east-1.neon.tech
NEON_POOLER_PORT=6432

# ===== OPCIÓN 2: Usar PostgreSQL Local (Desarrollo sin internet) =====
DB_TYPE=local
host=localhost
database=instacart_database
user=postgres
password=tu_contrasena_local
port=5432
```

**SEGURIDAD CRÍTICA**: 
- Nunca subir `.env` a Git (debe estar en `.gitignore`)
- Usar credenciales robustas (Neon genera automáticamente contraseñas fuertes)
- Regenerar credenciales cada 3 meses en Project Settings
- No compartir con otros desarrolladores (compartir solo `.env.example` sin valores)

---

## Connection Pooling en Neon

### ¿Por qué Connection Pooling es importante?

Neon utiliza **PgBouncer** (pooler nativo) para gestionar conexiones. Este es un concepto crítico:

#### Conexión Directa
- **Cuándo usar**: Scripts Python (data_ingestation.py), migraciones de BD, análisis de datos.
- **Ventaja**: Transacciones complejas y statements preparados permiten máximo rendimiento.
- **Limitación**: Máximo 100 conexiones simultáneas en plan free.
- **Host**: `ep-xxxxx.us-east-1.neon.tech` (sin `-pooler`)

#### Conexión Pooled
- **Cuándo usar**: APIs REST, aplicaciones web, miles de usuarios concurrentes.
- **Ventaja**: Reutilización de conexiones permite miles de usuarios con pocas conexiones de BD.
- **Limitación**: No soporta algunos statements preparados complejos (excepto en modo transaction).
- **Host**: `ep-xxxxx-pooler.us-east-1.neon.tech` (con `-pooler`)

### Comparación Visual

```
CONEXIÓN DIRECTA:
Aplicación 1 ──┐
Aplicación 2 ──┼──── Base de Datos
Aplicación 3 ──┘
(100 conexiones máximo)

CONEXIÓN POOLED (PgBouncer):
Aplicación 1 ──┐
Aplicación 2 ──┼──── PgBouncer (reutiliza connecciones)
Aplicación 3 ──┤
Aplicación ... ├──── 10-20 conexiones a BD actual
(Miles de users)
```

### Configuración Recomendada para Este Proyecto

**Para data_ingestation.py** (conexión directa):
```python
DB_CONFIG = {
    'host': os.getenv('NEON_HOST'),
    'database': os.getenv('NEON_DATABASE'),
    'user': os.getenv('NEON_USER'),
    'password': os.getenv('NEON_PASSWORD'),
    'port': int(os.getenv('NEON_PORT', 5432))
}
```

**Para futuras APIs** (conexión pooled):
```python
DB_CONFIG = {
    'host': os.getenv('NEON_POOLER_HOST'),
    'database': os.getenv('NEON_DATABASE'),
    'user': os.getenv('NEON_USER'),
    'password': os.getenv('NEON_PASSWORD'),
    'port': int(os.getenv('NEON_POOLER_PORT', 6432))
}
```

---

## Ejecución con Neon

### Paso 1: Verificar Conexión

```bash
# Verificar que las variables de entorno se cargaron correctamente
python -c "from dotenv import load_dotenv; import os; load_dotenv(); \
print(f'DB_TYPE: {os.getenv(\"DB_TYPE\")}'); \
print(f'Host: {os.getenv(\"NEON_HOST\")}'); \
print(f'Database: {os.getenv(\"NEON_DATABASE\")}')"
```

Salida esperada:
```
DB_TYPE: neon
Host: ep-xxxxx.us-east-1.neon.tech
Database: instacart_database
```

### Paso 2: Verificar Conectividad de Red

```bash
# Probar conectividad al host (opcional pero recomendado)
ping ep-xxxxx.us-east-1.neon.tech
```

Si el ping falla pero tienes internet, no te preocupes; es probable que Neon bloquee ICMP. Continúa al siguiente paso.

### Paso 3: Ejecutar el Script de Ingestación

```bash
# Con entorno virtual activado
source .venv/bin/activate

# Ejecutar ingestación
python src/data/data_ingestation.py
```

**Salida esperada:**
```
2026-03-13 10:30:15 - INFO - ============================================================
2026-03-13 10:30:15 - INFO - INICIO DE INGESTACIÓN DE DATOS
2026-03-13 10:30:15 - INFO - Tipo de BD: NEON (Serverless)
2026-03-13 10:30:15 - INFO - Host: ep-xxxxx.us-east-1.neon.tech
2026-03-13 10:30:15 - INFO - Database: instacart_database
2026-03-13 10:30:15 - INFO - ============================================================
2026-03-13 10:30:16 - INFO - Conexion exitosa
2026-03-13 10:30:17 - INFO - Generando 206209 usuarios
Generando Usuarios: 100%|██████████| 206209/206209
...
2026-03-13 10:40:45 - INFO - ============================================================
2026-03-13 10:40:45 - INFO - PROCESO COMPLETADO EXITOSAMENTE
2026-03-13 10:40:45 - INFO - ============================================================
```

**Tiempo estimado**: 10-20 minutos (más lento que local por latencia de red, pero aceptable)

### Paso 4: Monitorear Progreso

Los logs se guardan en:
```
reports/logs/data_ingestation.log
```

Visualizar en tiempo real:
```bash
tail -f reports/logs/data_ingestation.log
```

---

## Dashboard de Neon

### Navegación Principal

Una vez que la ingestación esté completa, accede a tu proyecto Neon en `neon.tech`.

#### 1. **SQL Editor**

En el panel izquierdo, hacer clic en **"SQL Editor"**:
- Ejecutar queries arbitrarias
- Ver resultados en tiempo real
- Descargar resultados en CSV

#### 2. **Tables**

Hacer clic en **"Tables"** para ver todas las tablas creadas:

```
instacart_database/
├── users_schema
│   └── users                          # ~206k registros
├── resources
│   ├── products                       # ~49k productos
│   ├── aisles                         # 134 aisles
│   └── departments                    # 21 departamentos
├── orders_schema
│   ├── orders                         # ~3.4M órdenes
│   ├── order_products_train           # ~32M items entrenamiento
│   └── order_products_prior           # ~34M items histórico
└── public
    └── [otras tablas]
```

#### 3. **Query Editor**

Ejemplo de queries útiles:

```sql
-- Contar registros por tabla
SELECT COUNT(*) as total_usuarios FROM users_schema.users;
SELECT COUNT(*) as total_productos FROM resources.products;
SELECT COUNT(*) as total_ordenes FROM orders_schema.orders;

-- Ver estadísticas de órdenes
SELECT 
    COUNT(*) as total_ordenes,
    AVG(days_since_prior_order) as dias_promedio_entre_compras,
    MIN(order_number) as primera_orden,
    MAX(order_number) as ultima_orden
FROM orders_schema.orders;

-- Ver tamaño de BD
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema', 'neon')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

#### 4. **Monitoring (Observability)**

Panel de **Monitoring** muestra:
- Uso de compute (vCPU %)
- Conexiones activas
- Operaciones por segundo
- Almacenamiento utilizado

**Nota**: En plan free, el compute se suspende tras 5 minutos sin actividad (eso es normal y reduce costos).

#### 5. **Database Branching**

Neon permite crear ramas de BD (feature única):

1. Ir a **Branches** en el panel izquierdo
2. Hacer clic en **"New Branch"**
3. Seleccionar rama base (generalmente `main`)
4. Nombrar la rama (ej: `feature/new-features`, `dev`, `test`)
5. Obtendrás un connection string diferente para esa rama

**Caso de uso**: Pruebas de nuevos scripts sin afectar la rama `main` (producción).

---

## Diferencias entre Conexión Directa y Pooled

### Cuándo Usar Cada Una

#### Conexión Directa (`ep-xxxxx.neon.tech`)

**Úsala para:**
- ✅ Scripts de data science (pandas, análisis)
- ✅ Migraciones de BD (Alembic, etc.)
- ✅ Bulk inserts (data_ingestation.py)
- ✅ Complex transactions
- ✅ Prepared statements complejos

**NO úsala para:**
- ❌ Aplicaciones web con miles de usuarios
- ❌ APIs REST de larga duración
- ❌ Algo que mantenga conexión permanente

#### Conexión Pooled (`ep-xxxxx-pooler.neon.tech`)

**Úsala para:**
- ✅ APIs REST
- ✅ Aplicaciones web
- ✅ Microservicios
- ✅ Anything con muchas conexiones concurrentes

**NO úsala para:**
- ❌ Long-running transactions (>1 min)
- ❌ Scripts que necesitan transaction state
- ❌ LISTEN/NOTIFY
- ❌ Prepared statements complejos (sí en transaction mode)

---

## Cold Starts en Neon

### ¿Qué es un Cold Start?

En plan free, Neon suspende el compute tras 5 minutos sin actividad. Cuando vuelves a conectar:

1. Neon reactiva el compute
2. Base de datos se reinicia
3. Primera query tarda 1-2 segundos adicionales (Cold Start)

**Esto es normal y totalmente aceptable para desarrollo/datos.**

### Ejemplo de Cold Start

```
FIRST QUERY AFTER 5 MIN INACTIVITY:
Query ejecutada 11:05:00 
├─ Cold start: ~1800 ms
├─ Query execution: ~50 ms
└─ Total: ~1850 ms

SUBSEQUENT QUERIES:
Query ejecutada 11:05:15
├─ Cold start: 0 ms (ya está warm)
├─ Query execution: ~50 ms
└─ Total: ~50 ms
```

### Cómo Evitar Cold Starts (Si es Crítico)

Si necesitas evitar cold starts:

1. **Upgradear a plan pagado** (Pro o superior)
2. **Usar cron jobs** para mantener BD activa (ping cada 4 minutos)
3. **Usar Database Branching** con rama `dev` para desarrollo

---

## Solución de Problemas

### Problema 1: Error de Autenticación

**Síntoma:**
```
psycopg2.OperationalError: FATAL: password authentication failed for user "neonuser"
```

**Causas y Soluciones:**

```bash
# 1. Verificar contraseña en .env
# Copiar exactamente del connection string en Neon Dashboard

# 2. Si contiene caracteres especiales (@, #, %, &, etc.):
# NO modificar, copiar tal cual en .env
# Neon maneja URL encoding automáticamente

# 3. Regenerar contraseña si la olvidaste:
# - Dashboard → Project Settings → Database
# - Hacer clic en usuario (a la derecha)
# - "Reset password"
```

---

### Problema 2: Conexión Rechazada / No Se Puede Conectar

**Síntoma:**
```
psycopg2.OperationalError: could not translate host name "ep-xxxxx.neon.tech"
```
o
```
Connection timeout
```

**Causas y Soluciones:**

**Causa 1: Sin conectividad a internet**
```bash
# Verificar conexión
ping google.com
ping 8.8.8.8

# Si fallan ambas, tu conexión a internet está offline
```

**Causa 2: Firewall bloqueando puerto 5432**
```bash
# Verificar conectividad a puerto 5432
nc -zv ep-xxxxx.neon.tech 5432

# Si falla, contactar a administrador de red
# Puerto 5432 debe estar permitido para salida
```

**Causa 3: VPN o proxy interfiriendo**
```bash
# Desconectar VPN temporalmente para probar
# Usar otro navegador/terminal si es posible
# Neon está bloqueado específicamente por tu firewall corporativo
```

**Causa 4: Host incorrecto en .env**
```bash
# ❌ Incorrecto
NEON_HOST=https://ep-xxxxx.neon.tech
NEON_HOST=ep-xxxxx.neon.tech:5432
NEON_HOST=ep-xxxxx.us-east-1.neon.tech:5432  # Puerto aquí NO

# ✅ Correcto
NEON_HOST=ep-xxxxx.us-east-1.neon.tech
NEON_PORT=5432  # En variable separada
```

---

### Problema 3: "Too Many Connections"

**Síntoma:**
```
psycopg2.OperationalError: too many connections for role "neonuser"
```

**Causa**: Plan free limita a 100 conexiones simultáneas (conexión directa).

**Soluciones**:

**Opción 1: Usar Connection Pooler (RECOMENDADO)**
```python
# Cambiar a conexión pooled para futuras APIs
DB_CONFIG = {
    'host': os.getenv('NEON_POOLER_HOST'),  # -pooler host
    'port': 6432,  # Puerto pooler
    # ... resto igual
}
```

**Opción 2: Implementar Connection Pooling en Python**
```python
from psycopg2 import pool

# En src/data/db_pool.py
connection_pool = psycopg2.pool.SimpleConnectionPool(
    1,      # Min conexiones
    10,     # Max conexiones
    host=os.getenv('NEON_HOST'),
    database=os.getenv('NEON_DATABASE'),
    user=os.getenv('NEON_USER'),
    password=os.getenv('NEON_PASSWORD'),
    port=int(os.getenv('NEON_PORT', 5432))
)

def get_connection():
    return connection_pool.getconn()

def return_connection(conn):
    connection_pool.putconn(conn)
```

**Opción 3: Reducir conexiones simultáneas**
- Cerrar conexiones explícitamente en código
- No usar `autocommit=True` indefinidamente
- Usar context managers para garantizar cierre

```python
import psycopg2

with psycopg2.connect(**db_config) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users_schema.users LIMIT 10")
        # Conexión se cierra automáticamente aquí
```

---

### Problema 4: Timeout en Ingestación

**Síntoma:**
```
psycopg2.OperationalError: server closed the connection unexpectedly
```
o
```
Timeout waiting for query response
```

**Causas y Soluciones**:

**Causa 1: Cold Start de Neon (1-2 seg después de 5 min inactivo)**
```python
# Solución: Simplemente esperar o reintentar
import time
import psycopg2

for attempt in range(3):
    try:
        conn = psycopg2.connect(**db_config)
        break
    except psycopg2.OperationalError as e:
        if attempt < 2:
            time.sleep(2)  # Esperar a que Neon se despierte
            continue
        raise
```

**Causa 2: Query muy larga**
```python
# Aumentar timeout en conexión
DB_CONFIG = {
    'host': host,
    'database': database,
    'user': user,
    'password': password,
    'port': port,
    'connect_timeout': 60,  # 60 segundos para conectar
    'options': '-c statement_timeout=600000'  # 10 minutos por query
}
```

**Causa 3: Network latency excesiva**
```bash
# Verificar latencia a Neon
ping ep-xxxxx.us-east-1.neon.tech

# Si la latencia es > 200ms, usar región más cercana
# Dashboard → Project Settings → Region
```

**Causa 4: Batch size demasiado grande**
```python
# Reducir page_size en execute_batch
from psycopg2.extras import execute_batch

# ❌ Puede timeout en conexión lenta
execute_batch(cursor, query, data, page_size=50000)

# ✅ Más seguro
execute_batch(cursor, query, data, page_size=5000)
```

---

### Problema 5: Almacenamiento Lleno (500 MB excedido)

**Síntoma:**
```
disk full or exceeded database size limitation
```

**Verificar uso actual:**
```sql
SELECT 
    pg_size_pretty(pg_database_size('instacart_database')) as db_size;
```

**Soluciones**:

**Opción 1: Limpiar datos no necesarios**
```sql
-- Eliminar órdenes antiguas (mantener solo período de interés)
DELETE FROM orders_schema.order_products_prior 
WHERE order_id IN (
    SELECT order_id FROM orders_schema.orders 
    WHERE order_date < '2014-01-01'  -- Ajustar según necesidad
);

-- Vacuum para liberar espacio
VACUUM FULL;
```

**Opción 2: Dividir ingestación en fases**
```bash
# En lugar de cargar todo a la vez:
python src/data/data_ingestation.py --year 2014 --month 6
python src/data/data_ingestation.py --year 2014 --month 7
# ... continuar mes a mes
```

**Opción 3: Usar Database Branching para pruebas**
```bash
# Crear rama test, no llenar rama main
# Dashboard → Branches → New Branch (rama de desarrollo)
```

**Opción 4: Upgradear plan (Pro)**
- Plan Pro: 8 GB almacenamiento
- 10x mayor que Free Tier
- Ideal si escalas en datos

---

### Problema 6: Queries Lentas

**Síntoma**: Queries que en local toman 100ms, en Neon tardan 5+ segundos.

**Causa**: Latencia de red + falta de índices.

**Soluciones**:

**1. Crear índices en tablas frecuentes**
```sql
-- Ver si existen índices
SELECT * FROM pg_stat_user_indexes;

-- Crear índices en columnas de búsqueda común
CREATE INDEX idx_orders_user_id ON orders_schema.orders(user_id);
CREATE INDEX idx_order_products_order_id ON orders_schema.order_products_train(order_id);
CREATE INDEX idx_products_department_id ON resources.products(department_id);
```

**2. Usar connection pooling para múltiples queries**
```python
# En lugar de múltiples conexiones separadas
conn = get_pooled_connection()
for query in queries:
    execute(conn, query)
return_connection(conn)
```

**3. Batch queries juntas**
```sql
-- En lugar de N queries separadas
BEGIN;
  INSERT INTO table1 VALUES (...);
  INSERT INTO table2 VALUES (...);
  UPDATE table3 SET ...;
COMMIT;
```

---

### Problema 7: Error "Project quota exceeded"

**Síntoma:**
```
Error: Project quota exceeded
```

**Causa**: Plan free permite solo 1 proyecto.

**Soluciones**:
- Usar Database Branching en el proyecto actual (limpio, no crea otros proyectos)
- Eliminar proyectos antiguos (si existen)
- Upgradear a Pro (permite múltiples proyectos)

---

## Gestión de Credenciales y Seguridad

### Regenerar Credenciales en Neon

Si crees que tu contraseña fue comprometida:

1. Dashboard → **Project Settings** (engranaje arriba-derecha)
2. Ir a **"Database"** en el sidebar
3. Buscar usuario `neonuser` en la lista
4. Hacer clic en el usuario y seleccionar **"Reset password"**
5. Copiar nueva contraseña
6. Actualizar `.env` con la nueva contraseña
7. Reiniciar aplicación/script

### Mejores Prácticas de Seguridad

### DO's
- ✅ Usar `.env` para credenciales
- ✅ Agregar `.env` a `.gitignore`
- ✅ Cambiar contraseña cada 3 meses
- ✅ Usar contraseñas de Neon (son seguras por defecto)
- ✅ Usar HTTPS/TLS (Neon lo hace automático)
- ✅ Limitar acceso a Database Branching para usuarios no técnicos
- ✅ Revisar Project Settings → Members regularmente

### DON'Ts
- ❌ Compartir credenciales por email/Slack
- ❌ Hardcodear credenciales en código
- ❌ Subir `.env` a Git
- ❌ Usar contraseña simple
- ❌ Usar mismo usuario para múltiples aplicaciones
- ❌ Dejar credenciales en historial de terminal (`history`)
- ❌ Pedir contraseña por teléfono/email

---

## Database Branching (Feature Único de Neon)

### ¿Qué es Database Branching?

Crea copias instantáneas de tu BD (similar a Git branches) sin duplicar almacenamiento.

**Casos de uso:**
- Develop features sin afectar `main`
- Probar migraciones antes de producción
- Crear ambientes de test

### Crear una Rama

1. Dashboard → **Branches** (lado izquierdo)
2. Hacer clic en **"New Branch"**
3. Seleccionar rama base (generalmente `main`)
4. Nombrar rama (ej: `feature/v2-models`, `test-data-cleanup`)
5. Esperar 30 segundos

**Cada rama tiene su propio connection string** (diferente host).

### Eliminar Una Rama

1. Ir a **Branches**
2. Hacer clic en el icono 🗑️ (trash) en la rama
3. Confirmar eliminación

---

## Monitoreo en Producción

### Dashboard de Monitoring (Neon)

Acceder a **"Monitoring"** en el proyecto:

- **Compute Time**: % de CPU utilizado
- **Active Connections**: Conexiones activas en tiempo real
- **Database Size**: Uso de almacenamiento actual
- **Operations/sec**: Queries ejecutadas por segundo

### SQL Queries de Monitoreo

```sql
-- Ver conexiones activas
SELECT * FROM pg_stat_activity WHERE datname = 'instacart_database';

-- Ver transacciones largas (> 5 minutos)
SELECT 
    pid,
    usename,
    query_start,
    now() - query_start as duration,
    query
FROM pg_stat_activity
WHERE state = 'active'
    AND query_start < now() - interval '5 minutes'
ORDER BY query_start DESC;

-- Ver índices sin usar
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;

-- Tamaño de tablas (para monitorear 500MB límite)
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as indexes_size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema', 'neon')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## Límites del Plan Gratuito vs Pro

| Característica | Free | Pro | Pro Plus | Custom |
|---|---|---|---|---|
| **Almacenamiento** | 500 MB | 8 GB | 50 GB | Ilimitado |
| **Compute** | Compartido (auto-suspend 5 min) | Dedicado | Dedicado | Dedicado |
| **Conexiones** | 100 directas, ∞ pooled | 1000 | Ilimitadas | Ilimitadas |
| **Proyectos** | 1 | 10 | Ilimitados | Ilimitados |
| **Ramas** | 1 | 10 | Ilimitadas | Ilimitadas |
| **Backups** | Diarios | Diarios | Bajo-demanda | Bajo-demanda |
| **Soporte** | Community | Email | Priority | Dedicated |
| **Precio** | $0 + $300 créditos | $25/mes | $350/mes | Custom |

**Free Tier Duración**: $300 créditos suelen durar 3-6 meses con uso moderado.

---

## Recursos Adicionales

### Documentación Oficial
- [Neon Documentation](https://neon.tech/docs)
- [Neon Connection Pooling](https://neon.tech/docs/connect/connection-pooling)
- [Neon Database Branching](https://neon.tech/docs/manage/branches)
- [Neon PgBouncer Config](https://neon.tech/docs/reference/compatibility-notes#pgbouncer)

### PostgreSQL
- [PostgreSQL 15 Docs](https://www.postgresql.org/docs/15/index.html)
- [psycopg2 Connection Parameters](https://www.psycopg.org/2/docs/module.html)
- [Query Performance Tuning](https://www.postgresql.org/docs/current/using-explain.html)

### Comunidad
- [Neon GitHub](https://github.com/neondatabase/neon)
- [Neon Community Forum](https://neon.tech/community)
- [Stack Overflow - Neon Tag](https://stackoverflow.com/questions/tagged/neon)

---

## Checklist de Setup en Neon

- [ ] Crear cuenta en neon.tech
- [ ] Crear proyecto `instacart-recsys`
- [ ] Obtener connection string del Dashboard
- [ ] Ejecutar SQL de creación de tablas (data/local_database/InstaCart_DataBase_Creation.sql)
- [ ] Verificar tablas en Table Editor de Neon
- [ ] Copiar parámetros en `.env`:
  - [ ] NEON_HOST
  - [ ] NEON_DATABASE
  - [ ] NEON_USER
  - [ ] NEON_PASSWORD
  - [ ] NEON_PORT
- [ ] Descargar CSVs de Kaggle
- [ ] Copiar a carpeta `data/raw/`
- [ ] Ejecutar `python src/data/data_ingestation.py`
- [ ] Verificar logs en `reports/logs/data_ingestation.log`
- [ ] Monitorear uso de almacenamiento en Neon Dashboard (debe estar < 500 MB)
- [ ] Ejecutar queries de validación en SQL Editor
- [ ] Revisar Project Settings → Members (si aplica)
- [ ] Hacer backup de credenciales en lugar seguro (password manager)

---

## Comparación: Workflow Local vs Neon

### Local
```
1. Instalar PostgreSQL (30 min)
   ↓
2. Crear BD localmente
   ↓
3. Ejecutar SQL
   ↓
4. Ejecutar ingestación con Python
   ↓
5. Verificar en pgAdmin
```

### Neon
```
1. Crear cuenta Neon (2 minutos)
   ↓
2. Crear proyecto (1 minuto)
   ↓
3. Ejecutar SQL en SQL Editor (30 segundos)
   ↓
4. Copiar connection string
   ↓
5. Actualizar .env
   ↓
6. Ejecutar ingestación con Python
   ↓
7. Verificar en Table Editor
```

**Total Neon**: 15-30 minutos (vs 45+ minutos con PostgreSQL local)

---

## Troubleshooting Rápido

| Problema | Síntoma | Solución Rápida |
|----------|---------|------------------|
| No conecta | `Connection refused` | Verificar `.env` y `NEON_HOST` |
| Auth falla | `Password authentication failed` | Copiar contraseña exacta de Dashboard |
| Too many connections | `too many connections` | Usar `-pooler` host o pool en código |
| Timeout | Query no responde en 30s | Aumentar timeout, crear índices |
| Sistema lento | Queries > 5s | Reducir batch size, usar índices |
| Almacenamiento lleno | `Disk full` | Limpiar datos viejos o upgradear |
| Cold start lento | +1-2s después de 5 min inactivo | Normal en free tier, upgradear si crítico |

---

## Migration desde Supabase a Neon (Si Aplica)

Si previamente usabas Supabase, migrar es simple:

```bash
# 1. En tu máquina local con Supabase conectado
pg_dump \
  -h [supabase_host] \
  -U [supabase_user] \
  -d [supabase_db] \
  > backup_supabase.sql

# 2. Conectar a Neon y restaurar
psql \
  -h [neon_host] \
  -U [neon_user] \
  -d [neon_database] \
  -f backup_supabase.sql
```

Luego actualizar credenciales en `.env` a Neon.

---

**Última actualización**: Marzo 13, 2026  
**Versión**: 1.0  
**Autor**: Equipo Técnico InstaCart RecSys  
**Estado**: Documentación oficial - Recomendado para producción
