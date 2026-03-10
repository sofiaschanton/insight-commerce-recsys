# 🚀 insight-commerce-recsys
Sistema de recomendación de próxima compra - Proyecto Final Data Science

---

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/sofiaschanton/insight-commerce-recsys.git
cd insight-commerce-recsys
```

### 2. Crear y activar un entorno virtual
```bash
# Crear entorno virtual con Python 3.10
py -3.10 -m venv venv

# Activar en Linux/macOS
source venv/bin/activate

# Activar en Windows
venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar las variables de entorno
Copia el archivo de ejemplo y editalo con tus datos:
```bash
cp .env.example .env
```

Luego abre `.env` y completa los valores según tu entorno (ver sección [Variables de entorno](#-variables-de-entorno)).

### 5. Ejecutar el ETL
```bash
python src/data/etl_dimensional.py
```

---

## 🔐 Variables de entorno

El proyecto utiliza un archivo `.env` en la raíz del proyecto para gestionar la configuración sensible. **Este archivo nunca debe subirse al repositorio.**

### Ejemplo de `.env`

```env
# Base de datos local (PostgreSQL)
LOCAL_HOST=localhost
LOCAL_DATABASE=InstaCart_DB
LOCAL_USER=postgres
LOCAL_PASSWORD=tu_password
LOCAL_PORT=5432

# Neon PostgreSQL (nube)
NEON_HOST=tu_host.neon.tech
NEON_DATABASE=neondb
NEON_USER=neondb_owner
NEON_PASSWORD=tu_password
NEON_PORT=5432
NEON_SSLMODE=require

# Configuración del proyecto
DATA_PATH=data/raw
RANDOM_SEED=42
N_USERS=100000
```

### Descripción de variables

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `LOCAL_HOST` | Dirección del servidor PostgreSQL local | `localhost` |
| `LOCAL_DATABASE` | Nombre de la base de datos local | `InstaCart_DB` |
| `LOCAL_USER` | Usuario PostgreSQL local | `postgres` |
| `LOCAL_PASSWORD` | Contraseña PostgreSQL local | — |
| `LOCAL_PORT` | Puerto PostgreSQL local | `5432` |
| `NEON_HOST` | Host de Neon PostgreSQL | — |
| `NEON_DATABASE` | Nombre de la base de datos en Neon | `neondb` |
| `NEON_USER` | Usuario Neon | — |
| `NEON_PASSWORD` | Contraseña Neon | — |
| `NEON_PORT` | Puerto Neon | `5432` |
| `NEON_SSLMODE` | Modo SSL Neon | `require` |
| `DATA_PATH` | Ruta a los CSVs originales | `data/raw` |
| `RANDOM_SEED` | Semilla aleatoria global | `42` |
| `N_USERS` | Usuarios a considerar en EDA local | `100000` |

> ⚠️ **Nunca compartas ni subas tu archivo `.env` a control de versiones.** Asegúrate de que `.env` esté incluido en tu `.gitignore`.

---

## 📁 Estructura del proyecto

```
insight-commerce-recsys/
│
├── data/
│   ├── raw/                        # CSVs originales — NO commiteados (.gitignore)
│   ├── processed/                  # Datos procesados para modelado — NO commiteados
│   ├── samples/                    # Muestras pequeñas para desarrollo y tests
│   └── local_database/
│       ├── InstaCart_DataBase_Creation_Relacional.sql
│       └── InstaCart_DataBase_Creation_Dimensional.sql
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_calidad_datos.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_lgbm_model.ipynb
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py
│   │   ├── etl_dimensional.py
│   │   ├── data_loader_supabase.py
│   │   └── validate.py
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── validation.py
│   ├── models/
│   │   ├── train.py
│   │   └── recommendation.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── api/
│       └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── reports/
│   ├── figures/
│   ├── logs/
│   └── informe_tecnico.md
│
├── models/                         # Modelos serializados — NO commiteados
│
├── tests/
│   └── test_api.py
│
├── docs/
│   ├── decisions.md
│   ├── feature_schema.md
│   ├── metricas_recomendacion.md
│   ├── arquitectura_deploy.md
│   ├── erd_dimensional.png
│   └── manual_usuario.md
│
├── .env.example
├── .gitignore
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## 🗄️ Base de datos

El proyecto usa dos bases de datos:

**Local (PostgreSQL):** modelo relacional normalizado con los CSVs originales de Instacart. Se usa como fuente para el ETL.

**Neon (PostgreSQL cloud):** modelo dimensional star schema con los datos filtrados y listos para feature engineering. Free tier con 0.5 GB.

### Esquema dimensional en Neon

| Tabla | Filas | Descripción |
|---|---|---|
| `dim_user` | ~10.000 | Usuarios aptos (≥5 órdenes prior + ≥1 orden train) |
| `dim_product` | ~26.000 | Productos aptos (≥50 compras globales) |
| `fact_order_products` | ~2.000.000 | Hechos de compra (prior + train) |

### Filtros aplicados en el ETL

- `eval_set != 'test'` — excluir órdenes de test
- Usuarios con ≥ 5 órdenes `prior` **Y** ≥ 1 orden `train`
- Productos con ≥ 50 compras globales en `prior`
- `LIMIT 10.000` usuarios aptos

---

## 🔀 Git Workflow — Ramas y Pull Requests

### 📐 Estructura de Ramas

```
main
└── develop
        ├── feature/eda-exploratorio
        ├── feature/feature-engineering
        ├── feature/etl-neon-dimensional
        ├── feature/modelo-lightgbm
        ├── feature/api-fastapi
        ├── feature/demo-streamlit
        ├── feature/dashboard-metricas
        └── hotfix/descripcion-del-fix
```

| Rama | Propósito | Desplegada en |
|---|---|---|
| `main` | Código en producción, siempre estable | 🟢 Producción |
| `develop` | Integración continua, base de trabajo | 🔵 Staging / QA |
| `feature/*` | Desarrollo de funcionalidades individuales | Local / Dev |

### 🔄 Flujo de Trabajo

#### 1. Crear una rama de feature
Siempre parte desde `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/nombre-descriptivo
```

**Convención de commits:**
```
tipo(scope): descripción breve en imperativo

Ejemplos:
feat(eda): agregar análisis de distribución de recompra por categoría
fix(etl): corregir filtro de usuarios en fact usando loaded_users desde Neon
docs(readme): actualizar instrucciones de instalación
refactor(model): separar pipeline de features en módulo independiente
test(api): agregar test de endpoint /recommend
chore(deps): actualizar lightgbm a versión 4.1

Tipos válidos: feat, fix, docs, refactor, test, chore, style, perf
```

#### 2. Desarrollar y hacer commits
```bash
git add .
git commit -m "feat: descripción clara del cambio"
git push origin feature/nombre-descriptivo
```

#### 3. Abrir un Pull Request hacia `develop`
- Ir al repositorio en GitHub
- Crear un PR desde `feature/*` → `develop`
- Completar la plantilla de PR
- Asignar al menos **un revisor** del equipo

#### 4. Revisión de código
- El revisor analiza el código, deja comentarios y aprueba o solicita cambios
- El autor responde los comentarios y realiza las correcciones necesarias
- **No se puede hacer merge sin al menos 1 aprobación**

#### 5. Merge a `develop`
Una vez aprobado, desde la interfaz de GitHub (squash merge recomendado).

#### 6. Release a `main`
Cuando `develop` está estable y validado en QA:
```bash
git checkout main
git pull origin main
git merge --no-ff develop
git tag -a v1.x.x -m "Release v1.x.x"
git push origin main --tags
```

### ✅ Reglas de Pull Requests

**Obligatorio para todo PR:**
- Al menos 1 aprobación de un miembro del equipo antes del merge
- Sin conflictos con la rama base
- Descripción clara de los cambios realizados

**Protecciones de ramas:**

| Rama | Merge directo | PR requerido | Aprobaciones mínimas |
|---|:---:|:---:|:---:|
| `main` | ❌ | ✅ | 1 |
| `develop` | ❌ | ✅ | 1 |
| `feature/*` | ✅ | — | — |

### 📝 Plantilla de Pull Request

```markdown
## 📋 Descripción
Breve resumen de los cambios y el contexto del problema que resuelven.

## 🔗 Issue relacionado
Card #NRO

## 🧪 Tipo de cambio
- [ ] ✨ Nueva funcionalidad
- [ ] 🐛 Corrección de bug
- [ ] ♻️ Refactor
- [ ] 📝 Documentación
- [ ] 🔧 Configuración / chore

## ✅ Checklist
- [ ] El código sigue los estándares del proyecto
- [ ] He añadido/actualizado tests necesarios
- [ ] He actualizado la documentación si aplica
- [ ] He probado los cambios localmente
- [ ] No hay conflictos con la rama base
```

---
