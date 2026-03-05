# Documentación: Configuración de Supabase - InstaCart RecSys

## Tabla de Contenidos
1. [¿Qué es Supabase?](#qué-es-supabase)
2. [Ventajas de Usar Supabase](#ventajas-de-usar-supabase)
3. [Setup en Supabase](#setup-en-supabase)
4. [Obtener Credenciales](#obtener-credenciales)
5. [Configuración del `.env`](#configuración-del-env)
6. [Ejecución con Supabase](#ejecución-con-supabase)
7. [Troubleshooting](#troubleshooting)

---

## ¿Qué es Supabase?

**Supabase** es una plataforma BaaS (Backend as a Service) de código abierto que proporciona:

- 🗄️ **PostgreSQL en la nube**: Base de datos relacional completamente funcional
- 🔐 **Autenticación integrada**: Sistema de usuarios y permisos
- 🔄 **Real-time**: Sincronización en tiempo real
- 📦 **Storage**: Almacenamiento de archivos
- 🎯 **API REST**: Acceso automático a datos

Para este proyecto usamos **Supabase solamente como PostgreSQL remoto**, sin necesidad de las otras características.

---

## Ventajas de Usar Supabase Vs. PostgreSQL Local

| Característica | Local | Supabase |
|----------------|-------|----------|
| **Instalación** | Requiere PostgreSQL instalado | Zero setup, creado en 2 min |
| **Mantenimiento** | Responsabilidad tuya | Gestionado por Supabase |
| **Backups** | Manual | Automático diario |
| **Acceso Remoto** | Requiere configuración | Automático y seguro |
| **Monitoreo** | Manual (tuyo) | Dashboard integrado |
| **Costo** | Libre (pero hardware tuyo) | Proyecto gratis hasta 500 MB |
| **Seguridad** | Depende de tu red | SSL/TLS automático |

### Plan Gratuito de Supabase

```
Proyecto Gratis (Ideal para desarrollo):
├─ Base de datos PostgreSQL
│  ├─ 500 MB de almacenamiento
│  ├─ CPU compartido
│  ├─ 2 conexiones simultáneas
│  └─ Backups diarios
├─ Autenticación: 100,000 usuarios
├─ Storage: 1 GB
└─ API calls: Ilimitadas
```

---

## Setup en Supabase

### Paso 1: Crear Cuenta

1. Ir a [supabase.com](https://supabase.com)
2. Hacer clic en **"Start your project"**
3. Crear cuenta con Google, GitHub o email

### Paso 2: Crear Proyecto

1. Ir a **"Projects"** en el dashboard
2. Hacer clic en **"New Project"**
3. Completar formulario:
   - **Project name**: `instacart-recsys` (o tu preferencia)
   - **Database password**: Crear contraseña fuerte (⚠️ Guardar!)
   - **Region**: Seleccionar región más cercana a ti
4. Esperar 1-2 minutos a que se cree

**IMPORTANTE**: Guardar la contraseña de base de datos en lugar seguro

### Paso 3: Crear Esquemas y Tablas

Una vez que el proyecto esté creado:

1. Ir a **SQL Editor** en el sidebar izquierdo
2. Hacer clic en **"New Query"**
3. Copiar y pegar el contenido de `InstaCart_DataBase_Creation.sql`
4. Ejecutar con botón **"RUN"** (o Ctrl+Enter)

**Nota**: Es importante mantener el nombre de la base de datos como `instacart_database`

---

## Obtener Credenciales

Las credenciales de Supabase están en tu dashboard:

### Método 1: Desde el Dashboard

1. Ir a **Project Settings** → **Database**
2. Encontrar **Connection Info**
3. Ver los parámetros de conexión

### Método 2: Desde SQL Editor

```sql
-- Ejecutar en SQL Editor para verificar conexión
SELECT current_user;
SELECT current_database();
```

### Parámetros a Obtener

```
HOST: [project-id].supabase.co
PORT: 5432
DATABASE: postgres (o instacart_database si ya existe)
USER: postgres
PASSWORD: [Tu contraseña]
```

**Formato de Host**: `xxxxx.supabase.co` (sin `https://` ni puertos)

**Seguridad**: Nunca compartir credenciales. Usar variables de entorno.

---

## Configuración del `.env`

Crear o editar el archivo `.env` en la raíz del proyecto:

```bash
# ===== OPCIÓN 1: Usar Supabase (Remoto) =====
DB_TYPE=supabase
SUPABASE_HOST=xxxxx.supabase.co
SUPABASE_DATABASE=postgres
SUPABASE_USER=postgres
SUPABASE_PASSWORD=tu_contraseña_fuerte_aqui
SUPABASE_PORT=5432

# ===== OPCIÓN 2: Usar PostgreSQL Local =====
DB_TYPE=local
host=localhost
database=instacart_database
user=postgres
password=tu_contraseña_local
port=5432
```

**SEGURIDAD**: 
- Nunca subir `.env` a Git (debe estar en `.gitignore`)
- Cambiar credenciales regularmente
- No compartir con otros

---

## Ejecución con Supabase

### Paso 1: Verificar Conexión

```bash
# Verificar que las variables de entorno se cargaron correctamente
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(f'DB_TYPE: {os.getenv(\"DB_TYPE\")}'); print(f'Host: {os.getenv(\"SUPABASE_HOST\")}')"
```

### Paso 2: Ejecutar el Script de Ingestación

```bash
python src/data/data_ingestation.py
```

**Salida esperada:**:
```
2026-03-05 14:30:15 - INFO - ============================================================
2026-03-05 14:30:15 - INFO - INICIO DE INGESTACIÓN DE DATOS
2026-03-05 14:30:15 - INFO - Tipo de BD: SUPABASE
2026-03-05 14:30:15 - INFO - Host: xxxxx.supabase.co
2026-03-05 14:30:15 - INFO - Database: postgres
2026-03-05 14:30:15 - INFO - ============================================================
2026-03-05 14:30:16 - INFO - Conexion exitosa
2026-03-05 14:30:17 - INFO - Generando 206209 usuarios
Generando Usuarios: 100%|██████████| 206209/206209
...
2026-03-05 14:40:45 - INFO - ============================================================
2026-03-05 14:40:45 - INFO - PROCESO COMPLETADO EXITOSAMENTE
2026-03-05 14:40:45 - INFO - ============================================================
```

### Paso 3: Monitorear Progreso

Los logs se guardan en:
```
reports/logs/data_ingestation.log
```

Visualizar en tiempo real:
```bash
tail -f reports/logs/data_ingestation.log
```

---

## Verificación en Supabase

### Desde el Dashboard

1. Ir a **"Table Editor"** en el sidebar
2. Ver todas las tablas populadas:
   - `users_schema.users`
   - `resources.products`
   - `resources.aisles`
   - `departments.departments`
   - `orders_schema.orders`
   - `orders_schema.order_products_train`
   - `orders_schema.order_products_prior`

### Desde SQL Editor

Ejecutar consultas para verificar:

```sql
-- Contar usuarios
SELECT COUNT(*) as total_usuarios FROM users_schema.users;

-- Ver ejemplos de productos
SELECT * FROM resources.products LIMIT 5;

-- Contar órdenes
SELECT COUNT(*) as total_ordenes FROM orders_schema.orders;
```

---

## Diferencias entre Local y Supabase

### Cuando Usar Cada Uno

#### 🖥️ PostgreSQL Local
**Usar cuando:**
- ✅ Desarrollo temprano
- ✅ Máquina potente disponible
- ✅ No necesitas acceso remoto
- ✅ Privacidad máxima en datos
- ✅ Sin conexión a internet

#### ☁️ Supabase
**Usar cuando:**
- ✅ Despliegue en producción
- ✅ Necesitas acceso remoto
- ✅ Múltiples desarrolladores
- ✅ Backups automáticos
- ✅ No quieres mantener servidor

### Migración: Local → Supabase

Si ya tienes datos en local y quieres migrar a Supabase:

```bash
# 1. Hacer dump de BD local
pg_dump -h localhost -U postgres instacart_database > backup.sql

# 2. Crear conexión a Supabase
psql -h xxxxx.supabase.co -U postgres -d postgres

# 3. Restaurar en Supabase
psql -h xxxxx.supabase.co -U postgres -d postgres -f backup.sql
```

---

## Solución de Problemas

### Problema: Error de Autenticación

**Síntoma:**
```
psycopg2.OperationalError: FATAL: role "postgres" does not exist
```

**Solución:**
```bash
# Verificar credenciales en .env
# El usuario por defecto es "postgres"
# Obtener contraseña de Project Settings → Database → Password
```

---

### Problema: Conexión Rechazada

**Síntoma:**
```
psycopg2.OperationalError: could not translate host name
```

**Causas y Soluciones:**

1. **Host incorrecto**:
```bash
# ❌ Incorrecto
SUPABASE_HOST=https://xxxxx.supabase.co
SUPABASE_HOST=xxxxx.supabase.co:5432

# ✅ Correcto
SUPABASE_HOST=xxxxx.supabase.co
```

2. **Sin internet**:
```bash
# Verificar conectividad
ping 8.8.8.8
ping google.com
```

3. **Firewall bloqueando**:
```bash
# Consultar con administrador de red
# Puerto 5432 debe estar permitido
```

---

### Problema: Timeout en Ingestación

**Síntoma**:
```
psycopg2.OperationalError: server closed the connection unexpectedly
```

**Soluciones**:

1. **Aumentar timeout en código** (en `data_ingestation.py`):
```python
DB_CONFIG = {
    'host': host,
    'database': database,
    'user': user,
    'password': password,
    'port': port,
    'connect_timeout': 60,  # Aumentar a 60 segundos
    'options': '-c statement_timeout=300000'  # 5 minutos por query
}
```

2. **Reducir tamaño de batch**:
```python
execute_batch(self.cursor, insert_query, users_data, page_size=5000)  # Reducir de 10000 a 5000
```

3. **Usar región más cercana**: Crear nuevo proyecto en región más próxima

---

### Problema: Límite de Conexiones

**Síntoma**:
```
too many connections for role "postgres"
```

**Causa**: Plan gratuito permite solo 2 conexiones simultáneas

**Solución**:
```python
# Usar connection pooling (agregar a data_ingestation.py)
from psycopg2 import pool

connection_pool = psycopg2.pool.SimpleConnectionPool(1, 1, **DB_CONFIG)
self.connection = connection_pool.getconn()
# ... usar conexión
connection_pool.putconn(self.connection)
```

---

### Problema: Base de Datos Llena

**Síntoma**:
```
disk full or exceeded database size limitation
```

**Causa**: Plan gratuito limitado a 500 MB

**Soluciones**:

1. **Dividir ingestación** en múltiples ejecuciones
2. **Upgradear a plan pagado**: Ir a Project Settings → Billing
3. **Limpiar datos antiguos**:
```sql
TRUNCATE TABLE orders_schema.order_products_train;
TRUNCATE TABLE orders_schema.order_products_prior;
-- Reiniciar secuencias si es necesario
```

---

## Gestión de Conexiones

### Pool de Conexiones (Recomendado para Producción)

Crear archivo `src/data/db_pool.py`:

```python
import psycopg2
from psycopg2 import pool
import os
from dotenv import load_dotenv

load_dotenv('.env')

DB_TYPE = os.getenv('DB_TYPE', 'local')

if DB_TYPE == 'supabase':
    config = {
        'host': os.getenv('SUPABASE_HOST'),
        'database': os.getenv('SUPABASE_DATABASE'),
        'user': os.getenv('SUPABASE_USER'),
        'password': os.getenv('SUPABASE_PASSWORD'),
        'port': os.getenv('SUPABASE_PORT', '5432'),
    }
else:
    config = {
        'host': os.getenv('host'),
        'database': os.getenv('database'),
        'user': os.getenv('user'),
        'password': os.getenv('password'),
        'port': os.getenv('port'),
    }

# Pool con máximo 5 conexiones
connection_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **config)

def get_connection():
    return connection_pool.getconn()

def return_connection(conn):
    connection_pool.putconn(conn)

def close_all():
    connection_pool.closeall()
```

---

## Consideraciones de Rendimiento

### Velocidad de Ingestación

**Local**: 3-5 minutos  
**Supabase (Plan Free)**: 15-30 minutos

**Por qué es más lento:**
- Conexión por internet (latencia)
- Recurso CPU compartido
- Limitaciones de ancho de banda

### Optimizaciones para Supabase

```python
# 1. Aumentar page_size de batches
execute_batch(cursor, query, data, page_size=5000)

# 2. Desactivar constraints durante carga
cursor.execute("ALTER TABLE ... DISABLE TRIGGER ALL")
# ... cargar datos
cursor.execute("ALTER TABLE ... ENABLE TRIGGER ALL")

# 3. Hacer commits menos frecuentes
# En lugar de commit() por cada tabla, hacer al final
```

---

## Monitoreo en Producción

### Dashboard de Supabase

1. **Database** → **Replication** (ver estado)
2. **Database** → **Backups** (ver backups automáticos)
3. **Security** → **API Keys** (gestionar acceso)

### Queries de Monitoreo

```sql
-- Ver conexiones activas
SELECT * FROM pg_stat_activity WHERE datname = 'postgres';

-- Ver tamaño de tablas
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Ver índices sin usar
SELECT * FROM pg_stat_user_indexes WHERE idx_scan = 0;
```

---

## Límites y Cuotas

### Plan Gratuito

```
Almacenamiento:     500 MB
Conexiones:         2 simultáneas
CPU:                Compartido
Backups:            Diarios (7 días)
Downtime:           SLA no garantizado
```

### Upgrades Disponibles

| Plan | Precio | Storage | Conexiones |
|------|--------|---------|------------|
| Free | $0/mes | 500 MB | 2 |
| Pro | $25/mes | 8 GB | 10 |
| Enterprise | Custom | Ilimitado | Ilimitadas |

---

## Mejores Prácticas

### DO's
- Usar SSL en conexiones (Supabase lo hace automático)
- Cambiar contraseña por defecto regularmente
- Usar `.env` para credenciales
- Hacer backups frecuentes
- Monitorear uso de almacenamiento
- Usar índices en queries frecuentes

### DON'Ts
- Compartir credenciales de BD
- Dejar usuario/contraseña por defecto
- Subir `.env` a Git
- Confiar en un único backup
- Hacer queries sin índices en datos grandes
- Mantener conectada permanentemente en pruebas

---

## Recursos Adicionales

- [Supabase Documentation](https://supabase.com/docs)
- [PostgreSQL Remote Connection](https://www.postgresql.org/docs/current/sql-syntax.html)
- [psycopg2 Connection Parameters](https://www.psycopg.org/2/docs/module.html#connection-factories)
- [Supabase Pricing](https://supabase.com/pricing)

---

## Checklist de Setup en Supabase

- [ ] Crear cuenta en supabase.com
- [ ] Crear nuevo proyecto
- [ ] Guardar contraseña de BD en lugar seguro
- [ ] Ejecutar script SQL de creación de tablas
- [ ] Obtener credenciales (Host, Usuario, Contraseña)
- [ ] Crear/actualizar archivo `.env`
- [ ] Descargar CSVs de Kaggle
- [ ] Ejecutar `python src/data/data_ingestation.py`
- [ ] Verificar datos en Table Editor
- [ ] Ejecutar queries de validación
- [ ] Revisar logs en `reports/logs/data_ingestation.log`

---

## Comparación: Workflow Local vs Supabase

### Local
```
1. Instalar PostgreSQL
   ↓
2. Crear BD
   ↓
3. Ejecutar SQL
   ↓
4. Ejecutar Python
   ↓
5. Verificar en pgAdmin
```

### Supabase
```
1. Crear cuenta (2 minutos)
   ↓
2. Crear proyecto (2 minutos)
   ↓
3. Ejecutar SQL en dashboard
   ↓
4. Copiar credenciales
   ↓
5. Ejecutar Python
   ↓
6. Verificar en Table Editor
```

**Total Supabase**: 20 minutos (vs 1+ hora con PostgreSQL local)

---

**Última actualización**: Marzo 5, 2026  
**Versión**: 1.0  
**Autor**: Equipo de Desarrollo InstaCart RecSys
